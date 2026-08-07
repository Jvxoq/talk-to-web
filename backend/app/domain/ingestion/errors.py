"""Business failures in the ingestion context."""


class IngestionError(Exception):
    """Base class for every failure the ingestion context can name."""


class UnsupportedDocumentType(IngestionError):
    def __init__(self, content_type: str | None) -> None:
        self.content_type = content_type
        super().__init__(f"Unsupported document type: {content_type or 'unknown'}")


class DocumentTooLarge(IngestionError):
    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        super().__init__(f"Document exceeds the {limit_bytes} byte limit")


class DocumentNotIndexable(IngestionError):
    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(f"No extractable text in {source}")
