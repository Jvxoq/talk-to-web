"""Unit tests for `LocalFileStorage`, on a real temporary directory."""

from collections.abc import AsyncIterator
from pathlib import Path

from app.adapters.storage.local_file_storage import LocalFileStorage
from app.domain.ingestion.value_objects import DocumentName


async def _stream(body: bytes) -> AsyncIterator[bytes]:
    yield body


def _read(reference: str) -> bytes:
    return Path(reference).read_bytes()


def _suffix_and_parent_ok(reference: str, root: Path) -> bool:
    return Path(reference).suffix == ".docx" and Path(reference).is_relative_to(
        root.resolve() / "1"
    )


class TestSameFilenameTwice:
    """Two uploads called `report.pdf` must not share one file on disk.

    `UploadDocument` stores the new file first and only then removes the
    document it replaces. When both carry the same name, a shared path means
    the removal deletes the file that was just uploaded, and indexing then
    fails to read it.
    """

    async def test_each_save_gets_its_own_reference(self, tmp_path: Path) -> None:
        storage = LocalFileStorage(tmp_path)
        name = DocumentName.sanitize("report.pdf")

        first = await storage.save(name, _stream(b"%PDF-old"), 1024, owner_id=1)
        second = await storage.save(name, _stream(b"%PDF-new"), 1024, owner_id=1)

        assert first != second
        assert _read(first) == b"%PDF-old"
        assert _read(second) == b"%PDF-new"

    async def test_deleting_the_old_upload_leaves_the_new_one(self, tmp_path: Path) -> None:
        storage = LocalFileStorage(tmp_path)
        name = DocumentName.sanitize("report.pdf")
        old = await storage.save(name, _stream(b"%PDF-old"), 1024, owner_id=1)
        new = await storage.save(name, _stream(b"%PDF-new"), 1024, owner_id=1)

        await storage.delete(old)

        assert _read(new) == b"%PDF-new"

    async def test_the_reference_keeps_the_suffix_the_extractor_routes_on(
        self, tmp_path: Path
    ) -> None:
        storage = LocalFileStorage(tmp_path)

        reference = await storage.save(
            DocumentName.sanitize("notes.docx"), _stream(b"PK\x03\x04"), 1024, owner_id=1
        )

        assert _suffix_and_parent_ok(reference, tmp_path)
