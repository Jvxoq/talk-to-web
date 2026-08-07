"""The one place domain failures turn into HTTP status codes.

Use cases raise business errors; they never import `HTTPException`. Everything
that decides "and what does the wire call that?" lives here.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.domain.chat.errors import (
    ChatError,
    ConversationNotFound,
    EmptyUserMessage,
    UnsupportedModel,
)
from app.domain.ingestion.errors import (
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

_INTERNAL_ERROR_DETAIL = "Internal server error"

_STATUS: dict[type[Exception], int] = {
    ConversationNotFound: status.HTTP_404_NOT_FOUND,
    EmptyUserMessage: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DocumentNotIndexable: status.HTTP_422_UNPROCESSABLE_CONTENT,
    UnsupportedDocumentType: status.HTTP_400_BAD_REQUEST,
    DocumentTooLarge: status.HTTP_413_CONTENT_TOO_LARGE,
    UnsupportedModel: status.HTTP_400_BAD_REQUEST,
    MalformedStartFrame: status.HTTP_400_BAD_REQUEST,
    TranscriptionUnavailable: status.HTTP_503_SERVICE_UNAVAILABLE,
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


def _to_response(exc: Exception) -> JSONResponse:
    code = _status_for(exc)

    if code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        # Exception messages at this level carry connection strings, API keys and
        # SQL. The traceback goes to the log; the client gets a fixed string.
        logger.exception("Unhandled {} while serving a request", type(exc).__name__)
        detail = _INTERNAL_ERROR_DETAIL
    else:
        detail = str(exc)

    return JSONResponse(
        status_code=code,
        content={"detail": detail, "type": type(exc).__name__},
    )


async def _handle(request: Request, exc: Exception) -> JSONResponse:
    return _to_response(exc)


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the domain error bases and a catch-all onto the app.

    `HTTPException` and `RequestValidationError` are deliberately left alone:
    FastAPI already handles them, and Starlette routes the catch-all `Exception`
    handler through its server-error middleware, which sits *outside* those, so
    registering one here does not shadow them.
    """
    for domain_base in (ChatError, IngestionError, TranscriptionError):
        app.add_exception_handler(domain_base, _handle)

    app.add_exception_handler(Exception, _handle)
