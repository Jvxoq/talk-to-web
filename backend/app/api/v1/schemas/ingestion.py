"""Wire shapes for the upload and document-manager endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.domain.ingestion.entities import UploadedDocument


class FileUploadResponse(BaseModel):
    message: str
    file_path: str
    # Returned so the client can remove what it just attached. Without it the
    # only way to undo an upload was to reload and find the row in a list.
    document_id: int


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    chunks_indexed: int
    created_at: datetime | None = None

    @classmethod
    def from_domain(cls, document: UploadedDocument) -> "DocumentOut":
        return cls.model_validate(document)
