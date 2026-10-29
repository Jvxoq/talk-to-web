"""Delivery-layer middleware. One concern per class, wired in `app/factory.py`."""

import re
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.context import REQUEST_ID_HEADER, reset_request_id, set_request_id

# An incoming id must match this, since it is echoed in headers and logs.
# A bad one is replaced, not rejected.
_WELL_FORMED = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_SCOPE_KEY = "request_id"


def request_id_of(request: Request) -> str | None:
    """This request's id, read from its scope.

    The catch-all error handler runs after the context variable is reset, so
    it reads the scope instead.
    """
    state: dict[str, Any] = request.scope.get("state", {})
    value = state.get(_SCOPE_KEY)
    return value if isinstance(value, str) else None


class RequestIdMiddleware:
    """Give every request an id, bind it to logs, and return it in a header.

    Raw ASGI, because `BaseHTTPMiddleware` buffers SSE and skips WebSockets.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request_id = self._incoming(scope) or uuid4().hex
        token = set_request_id(request_id)
        # Written to both places on purpose - see `request_id_of` above.
        scope.setdefault("state", {})[_SCOPE_KEY] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            # Reset, because the server's context outlives the request.
            reset_request_id(token)

    @staticmethod
    def _incoming(scope: Scope) -> str | None:
        """A well-formed id from the proxy or client, so one id spans every hop.

        It identifies a request only. Nothing is authorized on it.
        """

        candidate = Headers(scope=scope).get(REQUEST_ID_HEADER)
        if candidate is not None and _WELL_FORMED.match(candidate):
            return candidate
        return None
