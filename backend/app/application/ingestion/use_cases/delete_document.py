"""Remove an uploaded document: its vectors, its file, and its record."""

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.ports import DocumentLock, FileStorage, VectorIndex
from app.domain.ingestion.errors import DocumentNotFound


class DeleteDocument:
    """Delete an uploaded document. Raises `DocumentNotFound` if there is none.

    Vectors and file go first, so a failure leaves the row for a retry.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        index: VectorIndex,
        storage: FileStorage,
        lock: DocumentLock,
    ) -> None:
        self._uow_factory = uow_factory
        self._index = index
        self._storage = storage
        self._lock = lock

    async def __call__(self, document_id: int, owner_id: int) -> None:
        # Shares a lock with `IndexDocument`, so a delete during indexing leaves
        # no orphaned vectors. The whole body, row read included, is locked.
        async with self._lock.hold(document_id), self._uow_factory() as uow:
            document = await uow.documents.get(document_id, owner_id)
            if document is None:
                raise DocumentNotFound(document_id)

            await self._index.delete_document(document_id, owner_id)
            # Old URL-ingested rows have no stored file.

            if not document.reference.startswith(("http://", "https://")):
                await self._storage.delete(document.reference)

            await uow.documents.delete(document_id, owner_id)
            await uow.commit()
