"""Is Qdrant answering?"""

from qdrant_client import AsyncQdrantClient


class QdrantProbe:
    """Lists collections with the app's Qdrant client. Implements `ReadinessProbe`.

    Proves the host answers and the key works. Does not require the collection,
    which a new deployment does not have yet.
    """

    name = "qdrant"

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def check(self) -> None:
        await self._client.get_collections()
