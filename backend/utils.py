from api.schemas import TextModelRequest
from typing_extensions import AsyncIterator
from groq import AsyncGroq
from loguru import logger
import json
import os
import aiofiles
from aiofiles.os import makedirs
from fastapi import UploadFile

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50


def _frame(**payload: object) -> str:
    """
    Builds one SSE frame.

    The payload is JSON-encoded rather than written as bare text: a token can
    contain newlines (markdown headings, lists, tables all do), and a raw
    newline would terminate the frame early and be lost by the client.
    """
    return f"data: {json.dumps(payload)}\n\n"


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
        logger.debug(f"Groq request: model={model}, prompt_length={len(prompt)} chars")
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
        logger.debug("Groq stream started")

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield _frame(delta=content)
        yield _frame(done=True)

        logger.debug("Groq stream completed")

    except Exception as e:
        logger.warning(f"Failed to stream chunks. Error: {e}")
        yield _frame(error=str(e))

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
