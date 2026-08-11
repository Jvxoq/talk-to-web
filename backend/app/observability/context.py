"""The current request's correlation id.

A `ContextVar` rather than a parameter threaded through every call: the id is
ambient like the logger is, and making it an argument would mean every use case
and every adapter grew a field that has nothing to do with what it does.

Nothing here imports a framework, so a log line, a Sentry event and an adapter
can all reach the same id without the layering having an opinion about it. The
one place it is *set* is the middleware in `app/api/middleware.py`.
"""

from contextvars import ContextVar, Token

# The header the id travels on, in both directions. Named once here because the
# middleware writes it, CORS has to expose it, and the tests read it.
REQUEST_ID_HEADER = "X-Request-ID"

# What a log line shows when there is no request in scope - startup, shutdown,
# a background task. A placeholder rather than an empty column, so the field is
# always present and always aligned.
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
