"""Remove an uploaded document: its vectors, its file, and its record."""

from app.application.common.uow import UnitOfWorkFactory
from app.application.ingestion.ports import DocumentLock, FileStorage, VectorIndex
from app.domain.ingestion.errors import DocumentNotFound


class DeleteDocument:
    """
    Delete a document a person uploaded, failing loudly when there was
    nothing to delete.

    Vectors and the stored file are removed before the database row, not
    after: if either fails, the row - and the ability to retry - is still
    there. Deleting the row first and the vectors second would leave orphaned
    passages a retry could no longer find, since nothing would still name
    them as belonging to this owner.
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
        # The lock is what makes a delete that arrives while `IndexDocument` is
        # still running leave nothing behind: either this runs first and the
        # indexer finds no row when it re-checks inside the same lock, or the
        # indexer writes first and this cleans everything up after. Without
        # it, a delete landing mid-indexing removes what exists so far and
        # reports success - and the indexer then upserts every point anyway,
        # leaving vectors retrievable forever with no row to delete them by.
        # The whole body stays inside the lock: the row read must not be
        # answered from a moment before the indexer's writes.
        async with self._lock.hold(document_id), self._uow_factory() as uow:
            document = await uow.documents.get(document_id, owner_id)
            if document is None:
                raise DocumentNotFound(document_id)

            await self._index.delete_document(document_id, owner_id)
            # A document ingested by the removed URL path has its source URL as
            # `reference`, not a storage path - no file was ever written, so there
            # is nothing here for `FileStorage` to remove.
            if not document.reference.startswith(("http://", "https://")):
                await self._storage.delete(document.reference)

            await uow.documents.delete(document_id, owner_id)
            await uow.commit()
