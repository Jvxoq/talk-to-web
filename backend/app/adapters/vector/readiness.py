"""Is Qdrant answering?"""

from qdrant_client import AsyncQdrantClient


class QdrantProbe:
    """
    Lists collections through the application's own Qdrant client.

    Structurally satisfies `app.application.health.ports.ReadinessProbe`.

    `get_collections` is the cheapest call that proves both halves of what
    readiness means here: the host is reachable, and the API key is accepted. A
    rejected key is otherwise indistinguishable from a healthy deployment until
    the first retrieval fails.

    Deliberately *not* a check that the configured collection exists. It is
    created on first upload, so a brand-new deployment legitimately has none,
    and gating readiness on it would mean a fresh environment never passes its
    own rollout.
    """

    name = "qdrant"

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def check(self) -> None:
        await self._client.get_collections()
