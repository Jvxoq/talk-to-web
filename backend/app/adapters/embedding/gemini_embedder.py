"""Google Gemini text embeddings."""

from google import genai
from google.genai import types
from loguru import logger


class GeminiEmbedder:
    """
    Turns text into a vector with Gemini.

    Satisfies `app.application.ingestion.ports.Embedder`. The client is handed
    in already built, so nothing here reads an API key or opens a connection at
    import time.
    """

    def __init__(self, client: genai.Client, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed one piece of text, truncated by the provider if it is too long."""
        logger.debug(f"Embedding text ({len(text)} chars)")

        result = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self._dimensions),
        )

        # An empty response is a real failure mode (quota, safety filter, bad
        # model name). Naming it beats an IndexError three frames up the stack.
        if not result.embeddings or result.embeddings[0].values is None:
            raise RuntimeError(f"Embedding provider returned no vector for model {self._model}")

        return list(result.embeddings[0].values)
