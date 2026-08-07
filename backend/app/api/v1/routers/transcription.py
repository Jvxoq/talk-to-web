"""The speech-to-text WebSocket route.

WebSockets bypass the exception handlers registered in `app.api.errors` — there
is no response object to return — so failure handling is local to this handler.
"""

from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.api.dependencies import TranscribeStreamDep
from app.api.v1.websocket_transport import WebSocketTransport

router = APIRouter(tags=["transcription"])


@router.websocket("/ws/transcribe/")
async def transcribe(websocket: WebSocket, use_case: TranscribeStreamDep) -> None:
    await websocket.accept()
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
