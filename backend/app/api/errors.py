"""The one place domain failures turn into HTTP status codes.

Use cases raise business errors; they never import `HTTPException`. Everything
that decides "and what does the wire call that?" lives here.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.middleware import request_id_of
from app.domain.chat.errors import (
    ChatError,
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
    DocumentNotFound: status.HTTP_404_NOT_FOUND,
    EmptyUserMessage: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DocumentNotIndexable: status.HTTP_422_UNPROCESSABLE_CONTENT,
    UnsupportedDocumentType: status.HTTP_400_BAD_REQUEST,
    DocumentTooLarge: status.HTTP_413_CONTENT_TOO_LARGE,
    UnsupportedModel: status.HTTP_400_BAD_REQUEST,
    UnsafeUrl: status.HTTP_400_BAD_REQUEST,
    # 422, not 400: the request is syntactically well-formed JSON with a valid
    # message field - nothing about the wire format was wrong. Not 403 either:
    # that status asserts an authorization decision ("we know who you are and
    # the answer is no"), and a guardrail refusal is not about who is asking.
    UnsafeUserMessage: status.HTTP_422_UNPROCESSABLE_CONTENT,
    MalformedStartFrame: status.HTTP_400_BAD_REQUEST,
    TranscriptionUnavailable: status.HTTP_503_SERVICE_UNAVAILABLE,
    # 401 for all three: "who are you" was not answered. A 403 would mean the
    # opposite - we know who you are and the answer is still no.
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

    # A new subclass of a mapped error should inherit its parent's status rather
    # than silently degrade to a 500 the day someone adds one.
    for base in type(exc).__mro__:
        inherited = _STATUS.get(base)
        if inherited is not None:
            return inherited

    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _headers_for(exc: Exception, code: int) -> dict[str, str]:
    """The headers a status code is not allowed to arrive without.

    Both of these are part of what the status *means*, not decoration: a 401
    without `WWW-Authenticate` is malformed per RFC 9110, and a 429 without
    `Retry-After` tells a client to back off without saying for how long, which
    it will interpret as "immediately".
    """
    if code == status.HTTP_401_UNAUTHORIZED:
        return {"WWW-Authenticate": "Bearer"}
    if isinstance(exc, RateLimited):
        return {"Retry-After": str(exc.retry_after_seconds)}
    return {}


def _to_response(exc: Exception, request_id: str | None) -> JSONResponse:
    code = _status_for(exc)

    content: dict[str, object] = {"detail": str(exc), "type": type(exc).__name__}

    if code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        # Exception messages at this level carry connection strings, API keys and
        # SQL. The traceback goes to the log; the client gets a fixed string.
        #
        # The id is bound explicitly because this handler runs outside the
        # middleware that sets the context variable the logger normally reads.
        logger.bind(request_id=request_id or NO_REQUEST_ID).exception(
            "Unhandled {} while serving a request", type(exc).__name__
        )
        # Reported here rather than left to the SDK's ASGI integration: handling
        # the exception is what stops it propagating, so from Sentry's point of
        # view nothing went wrong unless we say so.
        report_exception(exc, request_id=request_id)
        content["detail"] = _INTERNAL_ERROR_DETAIL
        # The one thing a user can usefully quote back. It is on the response
        # header too, but nobody reporting a broken page has devtools open, and
        # the fixed detail above leaves them nothing else to give.
        content["request_id"] = request_id
    if isinstance(exc, RateLimited):
        # The same number as the `Retry-After` header, in the body, because the
        # header is the one thing a browser cannot read: `fetch` exposes only
        # the CORS-safelisted response headers unless the server opts each one
        # in, and the frontend is not always same-origin with the API. The
        # alternative - scraping the seconds back out of an English sentence -
        # breaks the first time the wording changes.
        content["retry_after_seconds"] = exc.retry_after_seconds

    headers = _headers_for(exc, code)
    if request_id is not None:
        # Set here as well as in the middleware: a response written by the
        # catch-all handler is sent by `ServerErrorMiddleware`, outside the
        # middleware that would otherwise stamp it, and a 500 is the response
        # most worth being able to correlate.
        headers[REQUEST_ID_HEADER] = request_id

    return JSONResponse(status_code=code, content=content, headers=headers)


async def _handle(request: Request, exc: Exception) -> JSONResponse:
    return _to_response(exc, request_id=request_id_of(request))


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the domain error bases and a catch-all onto the app.

    `HTTPException` and `RequestValidationError` are deliberately left alone:
    FastAPI already handles them, and Starlette routes the catch-all `Exception`
    handler through its server-error middleware, which sits *outside* those, so
    registering one here does not shadow them.
    """
    for domain_base in (ChatError, IngestionError, TranscriptionError, IdentityError, UsageError):
        app.add_exception_handler(domain_base, _handle)

    app.add_exception_handler(Exception, _handle)
