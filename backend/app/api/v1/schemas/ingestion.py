"""Wire shapes for the upload endpoint."""

from pydantic import BaseModel


class FileUploadResponse(BaseModel):
    message: str
    file_path: str
