"""Value objects for ingestion: the safe name of an upload, and a piece of a document."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.domain.ingestion.errors import UnsupportedDocumentType

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_STEM = 100


@dataclass(frozen=True, slots=True)
class DocumentName:
    """
    A filename that is safe to join onto a storage directory.

    A client-supplied name is untrusted input: "../../etc/passwd" and a name
    made entirely of separators both have to stop being paths before anything
    opens a file with them.
    """

    value: str

    @classmethod
    def sanitize(cls, raw: str | None) -> "DocumentName":
        candidate = PurePosixPath(raw or "").name.replace("\\", "/")
        candidate = PurePosixPath(candidate).name
        cleaned = _UNSAFE_CHARS.sub("_", candidate).lstrip(".")
        if not cleaned:
            raise UnsupportedDocumentType(raw)

        stem, dot, suffix = cleaned.rpartition(".")
        if not dot:
            return cls(cleaned[:_MAX_STEM])
        return cls(f"{stem[:_MAX_STEM]}.{suffix[:10]}")

    @property
    def stem(self) -> str:
        return self.value.rpartition(".")[0] or self.value

    def with_suffix(self, suffix: str) -> "DocumentName":
        return DocumentName(f"{self.stem}.{suffix.lstrip('.')}")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embeddable slice of a document, with the file it came from."""

    text: str
    source: str
