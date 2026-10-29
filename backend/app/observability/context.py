"""The current request's correlation id, in a `ContextVar`.

Any layer can read it. Only `app/api/middleware.py` sets it.
"""

from contextvars import ContextVar, Token

# The header the id travels on, in both directions.
REQUEST_ID_HEADER = "X-Request-ID"

# Shown in logs outside a request.

NO_REQUEST_ID = "-"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> Token[str | None]:
    """Bind an id to the current context. Returns the token that undoes it."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    """The id of the request being served, or None outside one."""
    return _request_id.get()
