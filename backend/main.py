from contextlib import asynccontextmanager
from typing import AsyncIterator, Annotated
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from dependencies import construct_prompt
from schemas import TextModelRequest, FileUploadResponse
from stream import WSConnectionManager, transcribe_audio_stream
from utils import stream_response, save_file
from rag.extractor import pdf_text_extractor
from rag.service import vector_service
from loguru import logger

# Initialse Lifespan for fastapi
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan function that initialises the LLM Client
    to serve requests
    """
    app.state.llm_client = AsyncGroq()

    yield


app = FastAPI(lifespan=lifespan)

ws_manager = WSConnectionManager()

app.mount("/pages", StaticFiles(directory="pages"), name="pages")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Generate Text Endpoint
@app.post("/generate/text/")
async def generate_text_stream_handler(
    body: TextModelRequest,
    prompt: str = Depends(construct_prompt)
) -> StreamingResponse:
    """
    POST endpoint for SSE streaming the generated tokens
    """
    try:
        return StreamingResponse(
            stream_response(
                app.state.llm_client, 
                body.model, body.temperature, 
                prompt
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.warning(f"Failed to stream response. Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to stream response")

@app.websocket("/ws/transcribe/")
async def transcribe_ws_handler(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that proxies browser microphone audio to Deepgram
    and streams transcripts back. The API key never leaves the server.
    """
    await ws_manager.connect(websocket)
    try:
        await transcribe_audio_stream(websocket, ws_manager)
    except WebSocketDisconnect:
        logger.debug("Client left the transcription socket")
    except Exception as e:
        logger.warning(f"Transcription stream failed. Error: {e}")
        try:
            await ws_manager.send({"type": "error", "detail": str(e)}, websocket)
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(websocket)


@app.post("/upload/file/")
async def file_upload_handler(
    file: Annotated[UploadFile, File(description="Uploaded pdf documents")],
    bg_text_processor: BackgroundTasks,
) -> FileUploadResponse:
    """
    Controller function to handle file uploads
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a PDF")
    try:
        filepath = await save_file(file)
        bg_text_processor.add_task(pdf_text_extractor, filepath)
        bg_text_processor.add_task(
            vector_service.store_file_contents_in_db,
            filepath.replace("pdf", "txt"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file. Error: {e}"
        )
    return FileUploadResponse(message="File uploaded successfully", file_path=filepath)