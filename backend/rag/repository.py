from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import ScoredPoint
from loguru import logger

# A Repository class that handles the CRUD operations of VectorDB
class VectorRepository():
    
    # Constructor that takes host and port values nad inits a async qdrant client
    def __init__(self, host: str, port: int) -> None:
        self.db_client = AsyncQdrantClient(host=host, port=port, check_compatibility=False)
    
    # Function to create collection
    async def create_collection(self, collection_name: str, size: int) -> bool:
        # Init the congif for collection
        vectors_config = models.VectorParams(
            size=size,
            distance=models.Distance.COSINE
        )
        # Check for duplicate collection
        response = await self.db_client.get_collections()
        # Delete duplicate collection
        collection_exits = any(collection.name == collection_name for collection in response.collections)
        if collection_exits:
            logger.debug(
                f"Collection {collection_name} already exists - recreating it"
            )
            await self.db_client.delete_collection(collection_name)
            return await self.db_client.create_collection(
                collection_name,
                vectors_config=vectors_config,
            )
        logger.debug(
            f"Creating a new collection {collection_name}"
        )
        # Create new collection with config
        return await self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
        )
    
    # Function to delete collection
    async def delete_collection(self, collection_name: str) -> bool:
        logger.debug(
            f"Deleted collection {collection_name}"
        )
        return await self.db_client.delete_collection(collection_name)
    
    # Function to create a point inside a collection
    async def create(
        self,
        collection_name: str,
        embedding_vector: list[float],
        original_text: str,
        source: str,
    ) -> None:
        response = await self.db_client.count(collection_name=collection_name)
        logger.debug(
            f"Creating new vector point with ID {response.count} in {collection_name}"
        )
        # Use upsert to create a point
        await self.db_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=response.count,
                    vector=embedding_vector,
                    payload={
                        "source": source,
                        "original_text": original_text,
                    }
                )
            ]
        )
    
    # Function for semantic search
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        retrieval_limit: int,
        score_threshold: float,
    ) -> list[ScoredPoint]:
        logger.debug(
            f"Searching for relevant items in the {collection_name} collection"
        )
        response = await self.db_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=retrieval_limit,
            score_threshold=score_threshold,
        )
        return response.points
