"""Upload routes: stream a file into a use case, then queue indexing."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status
from starlette.responses import Response

from app.api.dependencies import (
    CurrentUserDep,
    DeleteDocumentDep,
    IndexDocumentDep,
    ListDocumentsDep,
    UploadDocumentDep,
)
from app.api.v1.schemas.ingestion import DocumentOut, FileUploadResponse
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
    user: CurrentUserDep,
) -> FileUploadResponse:
    result = await upload_document(
        UploadDocumentInput(
            filename=file.filename,
            content_type=file.content_type,
            stream=_chunks(file, _CHUNK_BYTES),
            owner_id=user.user_id,
        )
    )

    # Fire-and-forget: BackgroundTasks dies with the process, has no retry and no
    # visibility. That is acceptable while a failed index only costs the user a
    # re-upload; the day indexing has to be guaranteed it belongs on a real queue.
    background.add_task(
        index_document, result.reference, result.name, result.document_id, user.user_id
    )

    return FileUploadResponse(
        message="File uploaded successfully",
        file_path=result.reference,
    )


@router.get("/documents/")
async def list_documents(
    use_case: ListDocumentsDep,
    user: CurrentUserDep,
) -> list[DocumentOut]:
    documents = await use_case(user.user_id)
    return [DocumentOut.from_domain(document) for document in documents]


# POST rather than DELETE, for the same reason as `/conversations/{id}/delete`:
# CORS on this app allows only GET, POST and OPTIONS across origins.
@router.post("/documents/{document_id}/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    use_case: DeleteDocumentDep,
    user: CurrentUserDep,
) -> Response:
    await use_case(document_id, user.user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
