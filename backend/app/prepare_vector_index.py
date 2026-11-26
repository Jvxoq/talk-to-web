"""Entry point that creates the Qdrant collection if it is missing. Not served by the API.

The vector store's counterpart to `alembic upgrade head`: it runs once per boot,
before the server starts, so the collection exists before any request needs it.
Without it the collection appeared only on the first upload, so after Qdrant
Cloud deleted an idle free cluster, every delete or search on the fresh cluster
failed until someone happened to upload a file.

`QdrantVectorIndex.ensure` never deletes and is safe to repeat, so a boot with
the collection already in place costs one listing call and three index calls.

Unlike Alembic, a failure here does not stop the boot. Postgres backs login and
every chat. Qdrant backs only documents. A suspended free cluster must not take
the whole app down. The upload path calls `ensure` again, and `/ready` reports
Qdrant as down, so the problem stays visible.

    uv run python -m app.prepare_vector_index
"""

import asyncio

from loguru import logger
from qdrant_client import AsyncQdrantClient

from app.adapters.vector.qdrant_index import QdrantVectorIndex
from app.observability.logging import configure_logging
from app.settings import get_settings


async def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.environment != "local")

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
        check_compatibility=False,
    )
    try:
        await QdrantVectorIndex(client=client, collection=settings.qdrant_collection).ensure(
            settings.embedding_dimensions
        )
    except RuntimeError as error:
        # `ensure` wraps every Qdrant failure in RuntimeError, so anything else
        # is a bug in this code and should still stop the boot.
        logger.opt(exception=error).error(
            f"Could not prepare collection {settings.qdrant_collection}; "
            f"starting without it: {error.__cause__!r}"
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
