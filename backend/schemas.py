from pydantic import BaseModel
from typing_extensions import Literal

class TextModelRequest(BaseModel):
    model: Literal["groq/compound", "llama-3.1-8b-instant"]
    user_input: str
    temperature: float = 0.0


class FileUploadResponse(BaseModel):
    message: str
    file_path: str