from typing import Any, AsyncGenerator
import re
import aiofiles
from google import genai

# Chunk Size
DEFAULT_CHUNK_SIZE = 1024 * 1024 * 1

# Embedding client
embed_client = genai.Client()
# async load
async def load(filepath: str) -> AsyncGenerator[str, Any]:
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        while chunk := await f.read(DEFAULT_CHUNK_SIZE):
            yield chunk
# clean
def clean(text: str) -> str:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    cleaned_text = t.replace("\n", " ").strip()
    return cleaned_text
# embed
async def embed(text: str) -> list[float]:
    result = await embed_client.aio.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values

# async def test():
#     async for chunk in load("/Users/jvxoq/Documents/Projects/fastapi/talk-to-the-web/backend/uploads/Jason_Daniel_Product_Intern.txt"):
#         print(await embed(clean(chunk)))

# asyncio.run(test())