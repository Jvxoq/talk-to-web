"""Upload route: stream the file into the use case, then queue indexing."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from app.api.dependencies import IndexDocumentDep, UploadDocumentDep
from app.api.v1.schemas.ingestion import FileUploadResponse
from app.application.ingestion.dto import UploadDocumentInput

router = APIRouter(tags=["ingestion"])

_CHUNK_BYTES = 1024 * 1024


async def _chunks(file: UploadFile, size: int) -> AsyncIterator[bytes]:
    """Read the upload in fixed slices so a large file never lands in memory whole."""
    while chunk := await file.read(size):
        yield chunk


@router.post("/upload/file/")
async def upload_file(
    file: Annotated[UploadFile, File()],
    upload_document: UploadDocumentDep,
    index_document: IndexDocumentDep,
    background: BackgroundTasks,
) -> FileUploadResponse:
    result = await upload_document(
        UploadDocumentInput(
            filename=file.filename,
            content_type=file.content_type,
            stream=_chunks(file, _CHUNK_BYTES),
        )
    )

    # Fire-and-forget: BackgroundTasks dies with the process, has no retry and no
    # visibility. That is acceptable while a failed index only costs the user a
    # re-upload; the day indexing has to be guaranteed it belongs on a real queue.
    background.add_task(index_document, result.reference, result.name)

    return FileUploadResponse(
        message="File uploaded successfully",
        file_path=result.reference,
    )
