"""The speech-to-text WebSocket route.

WebSockets bypass the exception handlers registered in `app.api.errors` — there
is no response object to return — so failure handling is local to this handler.
They also bypass CORS: a handshake is not a cross-origin fetch, there is no
preflight, and `CORSMiddleware` never sees it. Once the backend answers on a
public domain that would otherwise make this the one unauthenticated, uncapped
route in the app, and every accepted socket opens a billed Deepgram session.
Hence both checks below, which are this route's own job precisely because the
browser will not do either for us.

The access token rides in the `Sec-WebSocket-Protocol` header rather than the
URL: browsers give no way to set an `Authorization` header on a WebSocket
handshake, but they do let JavaScript choose the offered subprotocols, and
those never land in a server access log or a browser history entry the way a
query parameter does. The client offers two values - a fixed marker first, the
token second - and the server always accepts with the marker alone, so the
token itself is never echoed back into a response header either. Both checks
run *before* `accept()`, so a caller who fails either never reaches an open
socket and never costs a transcription session.
"""

from contextlib import suppress
from typing import Final

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.api.dependencies import IdentifyRequestDep, TranscribeStreamDep, WebSocketOriginsDep
from app.api.v1.websocket_transport import WebSocketTransport
from app.domain.identity.errors import IdentityError

router = APIRouter(tags=["transcription"])

# RFC 6455's "policy violation". Sent before `accept()`, which Starlette turns
# into an HTTP 403 on the handshake itself, so a rejected client never reaches
# an open socket and never costs a transcription session.
_POLICY_VIOLATION: Final = 1008

# The subprotocol the server always accepts with. Never the token itself - a
# `Sec-WebSocket-Protocol` response header is not a secret channel, it is just
# handshake negotiation, and only this fixed marker is safe to echo back.
_TOKEN_SUBPROTOCOL: Final = "access_token"  # noqa: S105 - a protocol name, not a credential


def _token_from_subprotocols(websocket: WebSocket) -> str:
    """Pull the second offered subprotocol out as the access token.

    The client offers `["access_token", "<token>"]`; a browser joins that list
    into one `Sec-WebSocket-Protocol` header with `, ` as the separator, so this
    undoes exactly that. Anything else offered - a missing header, a lone
    marker, extra entries - is treated as no token at all rather than guessed
    at, and falls through to the same rejection as an empty one.
    """
    header = websocket.headers.get("sec-websocket-protocol", "")
    offered = [part.strip() for part in header.split(",")]
    if len(offered) != 2 or offered[0] != _TOKEN_SUBPROTOCOL:
        return ""
    return offered[1]


@router.websocket("/ws/transcribe/")
async def transcribe(
    websocket: WebSocket,
    use_case: TranscribeStreamDep,
    allowed_origins: WebSocketOriginsDep,
    identify: IdentifyRequestDep,
) -> None:
    origin = websocket.headers.get("origin")
    if origin not in allowed_origins:
        # A missing Origin is refused along with a wrong one. Browsers always
        # send it on a WebSocket handshake, so the only clients this turns away
        # are the non-browser ones — which is the whole point, since the frontend
        # is the only legitimate caller.
        logger.warning(f"Rejected transcription handshake from origin {origin!r}")
        await websocket.close(code=_POLICY_VIOLATION, reason="Origin not allowed")
        return

    try:
        user = await identify(_token_from_subprotocols(websocket))
    except IdentityError as error:
        # `app.api.errors` never sees a WebSocket - there is no response to
        # attach a status to - so the refusal is spelled out here instead.
        logger.warning(f"Rejected transcription handshake: {error}")
        await websocket.close(code=_POLICY_VIOLATION, reason="Not authenticated")
        return

    logger.debug(f"Transcription socket opened for user {user.user_id}")
    await websocket.accept(subprotocol=_TOKEN_SUBPROTOCOL)
    transport = WebSocketTransport(websocket)

    try:
        await use_case(transport)
    except WebSocketDisconnect:
        # The ordinary ending: the user stopped recording or closed the tab.
        logger.debug("Transcription client disconnected")
    except Exception as exc:
        logger.opt(exception=exc).warning("Transcription session failed")
        # Best effort only — the socket is quite likely already gone, and a
        # failed send here would replace the real cause in the logs.
        with suppress(Exception):
            await transport.error("Transcription failed")
    finally:
        # Already-closed sockets raise RuntimeError rather than returning quietly.
        with suppress(RuntimeError):
            await websocket.close()
