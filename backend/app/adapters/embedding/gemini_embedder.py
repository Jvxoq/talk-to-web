"""Google Gemini text embeddings."""

from google import genai
from google.genai import types
from loguru import logger


class GeminiEmbedder:
    """Turns text into a vector with Gemini. Implements `Embedder`."""

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

        # Quota, safety filter or bad model name. Fail clearly here.

        if not result.embeddings or result.embeddings[0].values is None:
            raise RuntimeError(f"Embedding provider returned no vector for model {self._model}")

        return list(result.embeddings[0].values)
