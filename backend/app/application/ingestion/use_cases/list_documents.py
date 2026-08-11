"""List one owner's uploaded documents for the document manager panel."""

from app.application.common.uow import UnitOfWorkFactory
from app.domain.ingestion.entities import UploadedDocument


class ListDocuments:
    """Fetch every document this owner has uploaded, newest first."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def __call__(self, owner_id: int) -> list[UploadedDocument]:
        async with self._uow_factory() as uow:
            return await uow.documents.list_by_owner(owner_id)
