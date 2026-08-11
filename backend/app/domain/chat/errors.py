"""Business failures in the chat context. No HTTP status codes live here."""


class ChatError(Exception):
    """Base class for every failure the chat context can name."""


class EmptyUserMessage(ChatError):
    def __init__(self) -> None:
        super().__init__("A user message cannot be empty")


class ConversationNotFound(ChatError):
    def __init__(self, conversation_id: int) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"Conversation {conversation_id} not found")


class UnsupportedModel(ChatError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model {model!r} is not available")


class UnsafeUrl(ChatError):
    """A URL the server must not open a connection to.

    Anyone can put a link in a chat message, and the server fetches it with the
    network position of the server - inside the VPC, next to the database, and
    one hop from the cloud metadata endpoint. A refused URL is named here so the
    reason survives into the log.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"Refusing to fetch {url}: {reason}")
