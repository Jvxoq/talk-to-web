import os
from config import QDRANT_HOST, QDRANT_PORT
from .repository import VectorRepository
from loguru import logger
from .transformer import embed, load, clean

# Vector service class using the VectorRepository
class VectorService(VectorRepository):
    """
    This class is supposed to make use of the VectorRepo
    to implement create, read and updata functions for the vector storage and retrieval
    operations

    1. Create the VectorService class by inheriting the VectorRepository class so 
    that you can use and extend common database operation methods.

    2. Use the store_file_content_in_db service method to asynchronously load, transform, and store raw text documents into the database in chunks.

    3. Use an asynchronous generator load() to load text chunks from a file 
    asynchronously.

    4. Create an instance of the VectorService to import and use across the 
    application.
    """
    def __init__(self, host: str, port: int) -> None:
        super().__init__(host, port)
    
    async def store_file_contents_in_db(
        self, 
        filepath: str,
        collection_name: str = "knowledge_base",
        collection_size: int = 512,
    ) -> None:
        """
        This function uses the parent class's function to create a collection
        and store the contents from the file at the filepath in the vector db
        """
        await self.create_collection(
            collection_name,
            collection_size,
        )
        logger.debug(f"Inserting {filepath} content into database")
        async for chunk in load(filepath): 
            logger.debug(f"Inserting '{chunk[0:20]}...' into database")

            embedding_vector = await embed(clean(chunk))
            filename = os.path.basename(filepath)
            await self.create(
                collection_name, embedding_vector, chunk, filename
            )

vector_service = VectorService(QDRANT_HOST, QDRANT_PORT)