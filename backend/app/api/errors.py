"""The one place domain errors become HTTP status codes.

Use cases raise domain errors and never import `HTTPException`.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.middleware import request_id_of
from app.domain.chat.errors import (
    ChatError,
    ConversationLimitReached,
    ConversationNotFound,
    EmptyUserMessage,
    UnsafeUrl,
    UnsafeUserMessage,
    UnsupportedModel,
)
from app.domain.identity.errors import (
    EmailAlreadyRegistered,
    IdentityError,
    InvalidCredentials,
    InvalidEmail,
    InvalidToken,
    TokenExpired,
    WeakPassword,
)
from app.domain.ingestion.errors import (
    DocumentNotFound,
    DocumentNotIndexable,
    DocumentTooLarge,
    IngestionError,
    UnknownConversation,
    UnsupportedDocumentType,
)
from app.domain.transcription.errors import (
    MalformedStartFrame,
    TranscriptionError,
    TranscriptionUnavailable,
)
from app.domain.usage.errors import RateLimited, UsageError
from app.observability.context import NO_REQUEST_ID, REQUEST_ID_HEADER
from app.observability.sentry import report_exception

_INTERNAL_ERROR_DETAIL = "Internal server error"

_STATUS: dict[type[Exception], int] = {
    ConversationNotFound: status.HTTP_404_NOT_FOUND,
    # 409, not 429: waiting does not help. The user must delete a conversation.
    ConversationLimitReached: status.HTTP_409_CONFLICT,
    DocumentNotFound: status.HTTP_404_NOT_FOUND,
    UnknownConversation: status.HTTP_404_NOT_FOUND,
    EmptyUserMessage: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DocumentNotIndexable: status.HTTP_422_UNPROCESSABLE_CONTENT,
    UnsupportedDocumentType: status.HTTP_400_BAD_REQUEST,
    DocumentTooLarge: status.HTTP_413_CONTENT_TOO_LARGE,
    UnsupportedModel: status.HTTP_400_BAD_REQUEST,
    UnsafeUrl: status.HTTP_400_BAD_REQUEST,
    # 422: the JSON is valid, and the refusal is not about who is asking (403).
    UnsafeUserMessage: status.HTTP_422_UNPROCESSABLE_CONTENT,
    MalformedStartFrame: status.HTTP_400_BAD_REQUEST,
    TranscriptionUnavailable: status.HTTP_503_SERVICE_UNAVAILABLE,
    # 401: identity was not established. 403 would mean known but refused.
    InvalidCredentials: status.HTTP_401_UNAUTHORIZED,
    InvalidToken: status.HTTP_401_UNAUTHORIZED,
    TokenExpired: status.HTTP_401_UNAUTHORIZED,
    EmailAlreadyRegistered: status.HTTP_409_CONFLICT,
    InvalidEmail: status.HTTP_422_UNPROCESSABLE_CONTENT,
    WeakPassword: status.HTTP_422_UNPROCESSABLE_CONTENT,
    RateLimited: status.HTTP_429_TOO_MANY_REQUESTS,
}


def _status_for(exc: Exception) -> int:
    """Exact type first, then the MRO, then 500."""
    direct = _STATUS.get(type(exc))
    if direct is not None:
        return direct

    # A new subclass inherits its parent's status instead of becoming a 500.
    for base in type(exc).__mro__:
        inherited = _STATUS.get(base)
        if inherited is not None:
            return inherited

    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _headers_for(exc: Exception, code: int) -> dict[str, str]:
    """Headers a status needs: `WWW-Authenticate` on 401, `Retry-After` on 429."""
    if code == status.HTTP_401_UNAUTHORIZED:
        return {"WWW-Authenticate": "Bearer"}
    if isinstance(exc, RateLimited):
        return {"Retry-After": str(exc.retry_after_seconds)}
    return {}


def _to_response(exc: Exception, request_id: str | None) -> JSONResponse:
    code = _status_for(exc)

    content: dict[str, object] = {"detail": str(exc), "type": type(exc).__name__}

    if code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        # 5xx messages can hold secrets and SQL. Log the traceback, send a fixed
        # string. The id is bound by hand: this runs outside the middleware.
        logger.bind(request_id=request_id or NO_REQUEST_ID).exception(
            "Unhandled {} while serving a request", type(exc).__name__
        )
        # Handled exceptions do not reach Sentry by themselves, so report it.
        report_exception(exc, request_id=request_id)
        content["detail"] = _INTERNAL_ERROR_DETAIL
        # In the body too, so a user can quote it without devtools.
        content["request_id"] = request_id
    if isinstance(exc, RateLimited):
        # Repeats `Retry-After` in the body, because cross-origin `fetch`
        # cannot read that header.
        content["retry_after_seconds"] = exc.retry_after_seconds

    headers = _headers_for(exc, code)
    if request_id is not None:
        # Set here too: a 500 is sent outside the middleware that stamps it.
        headers[REQUEST_ID_HEADER] = request_id

    return JSONResponse(status_code=code, content=content, headers=headers)


async def _handle(request: Request, exc: Exception) -> JSONResponse:
    return _to_response(exc, request_id=request_id_of(request))


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for the domain error bases and a catch-all.

    The catch-all does not shadow FastAPI's own `HTTPException` handling.
    """

    for domain_base in (ChatError, IngestionError, TranscriptionError, IdentityError, UsageError):
        app.add_exception_handler(domain_base, _handle)

    app.add_exception_handler(Exception, _handle)
