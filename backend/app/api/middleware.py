"""Delivery-layer middleware. One concern per class, wired in `app/main.py`."""

import re
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.context import REQUEST_ID_HEADER, reset_request_id, set_request_id

# What an id from upstream is allowed to look like before we adopt it. The value
# is echoed back in a response header and written into every log line for the
# request, so an unvalidated one is a header-injection and log-forging primitive
# handed to whoever calls us. Anything failing this is replaced, not rejected:
# a malformed correlation id is not worth a failed request.
_WELL_FORMED = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_SCOPE_KEY = "request_id"


def request_id_of(request: Request) -> str | None:
    """The id `RequestIdMiddleware` gave this request, read off its scope.

    The context variable is the normal way to reach it, but not the only one it
    needs: the catch-all exception handler runs inside Starlette's
    `ServerErrorMiddleware`, which is built *outside* every user middleware, so
    by the time it is called this middleware's `finally` has already unset the
    variable. The scope is the one thing the two frames still share.
    """
    state: dict[str, Any] = request.scope.get("state", {})
    value = state.get(_SCOPE_KEY)
    return value if isinstance(value, str) else None


class RequestIdMiddleware:
    """Give every request an id, log-bind it, and send it back to the caller.

    A raw ASGI middleware rather than `BaseHTTPMiddleware` for two reasons: that
    base class buffers through an anyio stream, which is the wrong shape for the
    SSE endpoint, and it does not run on a WebSocket at all - and a dropped
    transcription socket is exactly the thing worth correlating.
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
            # The id belongs to this request and this task. Resetting matters
            # for the server's own context, which outlives the call.
            reset_request_id(token)

    @staticmethod
    def _incoming(scope: Scope) -> str | None:
        """An id from the proxy or the client, if it is one we can safely reuse.

        Trusting the header is the point: it is what lets one id span Caddy's
        access log, this service and whatever the frontend reports. Nothing is
        authorized on it - it identifies a request, never a caller.
        """
        candidate = Headers(scope=scope).get(REQUEST_ID_HEADER)
        if candidate is not None and _WELL_FORMED.match(candidate):
            return candidate
        return None
