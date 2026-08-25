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


class DocumentNotFound(IngestionError):
    def __init__(self, document_id: int) -> None:
        self.document_id = document_id
        super().__init__(f"Document {document_id} not found")


class UnknownConversation(IngestionError):
    """The thread an upload named is not one this account owns.

    Checked rather than trusted. A document filed under a stranger's thread is
    already unreadable - every search matches owner and conversation together -
    but writing it at all lets one account put rows and vectors into another's
    space, and the storage bill follows the wrong person.
    """

    def __init__(self, conversation_id: int) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"Conversation {conversation_id} not found")
