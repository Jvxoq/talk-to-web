"""The speech-to-text WebSocket route.

WebSockets skip `app.api.errors` and CORS, so this route checks Origin and
auth itself, before `accept()`. Each accepted socket costs a Deepgram session.

Browsers cannot set `Authorization` on a WebSocket, so the token is the second
offered subprotocol. That keeps it out of URLs and logs. The server accepts
with the fixed marker only, so the token is never echoed back.
"""

from contextlib import suppress
from typing import Final

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.api.dependencies import IdentifyRequestDep, TranscribeStreamDep, WebSocketOriginsDep
from app.api.v1.websocket_transport import WebSocketTransport
from app.domain.identity.errors import IdentityError
from app.domain.usage.errors import RateLimited

router = APIRouter(tags=["transcription"])

# "Policy violation". Before `accept()`, Starlette sends it as a 403.
_POLICY_VIOLATION: Final = 1008

# The fixed marker the server accepts with. Never the token.
_TOKEN_SUBPROTOCOL: Final = "access_token"  # noqa: S105 - a protocol name, not a credential


def _token_from_subprotocols(websocket: WebSocket) -> str:
    """Read the token from `["access_token", "<token>"]`. Any other shape gives ""."""
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
        # A missing Origin is refused too. Browsers always send one.
        logger.warning(f"Rejected transcription handshake from origin {origin!r}")
        await websocket.close(code=_POLICY_VIOLATION, reason="Origin not allowed")
        return

    try:
        user = await identify(_token_from_subprotocols(websocket))
    except IdentityError as error:
        # `app.api.errors` never sees WebSockets, so refuse here.
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
    except RateLimited:
        # Daily budget spent. Tell the client why.
        logger.warning("Rejected transcription session: daily call budget spent")
        with suppress(Exception):
            await transport.error("Service is busy right now - please try again later.")
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=exc).warning("Transcription session failed")
        # Best effort: the socket is probably gone already.

        with suppress(Exception):
            await transport.error("Transcription failed")
    finally:
        # Already-closed sockets raise RuntimeError rather than returning quietly.
        with suppress(RuntimeError):
            await websocket.close()
