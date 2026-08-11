"""Wire shapes for the upload and document-manager endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.domain.ingestion.entities import UploadedDocument


class FileUploadResponse(BaseModel):
    message: str
    file_path: str


class UrlUploadRequest(BaseModel):
    url: str


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    chunks_indexed: int
    created_at: datetime | None = None

    @classmethod
    def from_domain(cls, document: UploadedDocument) -> "DocumentOut":
        return cls.model_validate(document)
