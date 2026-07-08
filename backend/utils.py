from schemas import TextModelRequest
from typing_extensions import AsyncIterator
from groq import AsyncGroq
from loguru import logger
import os
import aiofiles
from aiofiles.os import makedirs
from fastapi import UploadFile

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50

# function to stream the response from the llm
async def stream_response(
    llm_client: AsyncGroq,
    model: str,
    temperature: float,
    prompt: str,
) -> AsyncIterator[str]:
    """
    A function that send's async post request to the groq llm,
    returns the response and handle errors
    """
    try:
        stream = await llm_client.chat.completions.create(
            # Required parameters
            messages=[
                # Set an optional system message. This sets the behavior of the
                # assistant and can be used to provide specific instructions for
                # how it should behave throughout the conversation.
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                # Set a user message for the assistant to respond to.
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # The language model which will generate the completion.
            model=model,
            # Optional Parameters
            # Controls randomness: lowering results in less random completions.
            temperature=temperature,
            # The maximum number of tokens to generate.
            max_completion_tokens=2048,
            # If set, partial message deltas will be sent.
            stream=True,
        )

        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    except Exception as e:
        logger.warning(f"Failed to stream chunks. Error: {e}")
        yield f"\n[error: {e}]"

# Function to save file asynchronously
async def save_file(file: UploadFile) -> str:
    """
    Asynchronously saves a file to the local filesystem.
    """
    await makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    async with aiofiles.open(filepath, "wb") as f:
        while chunk := await file.read(DEFAULT_CHUNK_SIZE):
            await f.write(chunk)
    return filepath
