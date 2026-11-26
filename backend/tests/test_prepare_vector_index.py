"""The boot step that creates the Qdrant collection, against in-process Qdrant.

The case it exists for: Qdrant Cloud deleted an idle free cluster, the new one
is empty, and the first request after the deploy is a delete or a search, not
an upload.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import PointStruct

from app import prepare_vector_index
from app.settings import Settings
from tests.conftest import logged

if TYPE_CHECKING:
    from loguru import Record

COLLECTION = "documents"
DIMENSIONS = 4


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncQdrantClient]:
    """One in-memory client shared by every boot, standing in for the cloud cluster."""
    shared = AsyncQdrantClient(location=":memory:")
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url="postgresql+psycopg://test:test@localhost/test",
        qdrant_collection=COLLECTION,
        embedding_dimensions=DIMENSIONS,
        llm_api_key=SecretStr("test"),
        tavily_api_key=SecretStr("test"),
        gemini_api_key=SecretStr("test"),
        deepgram_api_key=SecretStr("test"),
        jwt_secret=SecretStr("test"),
    )
    monkeypatch.setattr(prepare_vector_index, "get_settings", lambda: settings)
    # It removes every loguru sink, including the one `log_records` adds.
    monkeypatch.setattr(prepare_vector_index, "configure_logging", lambda **_: None)

    def build(**_: Any) -> AsyncQdrantClient:
        return shared

    monkeypatch.setattr(prepare_vector_index, "AsyncQdrantClient", build)
    # `main` closes the client it built; closing the in-memory one would wipe
    # the state the assertions read, so the close is a no-op here.
    close = shared.close
    monkeypatch.setattr(shared, "close", _noop)
    yield shared
    await close()


async def _noop(**_: Any) -> None:
    return None


async def test_creates_the_missing_collection_with_the_embedding_size(
    client: AsyncQdrantClient,
) -> None:
    await prepare_vector_index.main()

    info = await client.get_collection(COLLECTION)
    assert info.config.params.vectors.size == DIMENSIONS  # type: ignore[union-attr]


async def test_second_boot_keeps_existing_points(client: AsyncQdrantClient) -> None:
    await prepare_vector_index.main()
    await client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={})],
    )

    await prepare_vector_index.main()

    assert (await client.count(COLLECTION)).count == 1


async def test_unreachable_qdrant_is_logged_and_does_not_stop_the_boot(
    client: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch, log_records: list["Record"]
) -> None:
    # A suspended free cluster: no reply at all. Login and chat do not need
    # Qdrant, so the server must still start.
    async def get_collections() -> None:
        raise ResponseHandlingException(httpx.ConnectError(""))

    monkeypatch.setattr(client, "get_collections", get_collections)

    await prepare_vector_index.main()

    # The adapter logs its own failure first; the boot step's line is last.
    level, message, has_traceback = logged(log_records)[-1]
    assert level == "ERROR"
    assert "starting without it" in message
    assert "ConnectError" in message
    assert has_traceback


async def test_a_bug_in_the_step_still_stops_the_boot(
    client: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def get_collections() -> None:
        raise TypeError("not a Qdrant failure")

    monkeypatch.setattr(client, "get_collections", get_collections)

    with pytest.raises(TypeError):
        await prepare_vector_index.main()
