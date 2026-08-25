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


class ConversationLimitReached(ChatError):
    """The account already holds as many conversations as it may.

    A cap rather than an eviction: deleting someone's oldest thread to make
    room for a new one destroys data they never asked to lose. Refusing tells
    them what to do about it.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"You can keep {limit} conversations at a time. Delete one to start another."
        )


class UnsupportedModel(ChatError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model {model!r} is not available")


class UnsafeUserMessage(ChatError):
    """A message a guardrail refused to send to the model.

    The message this carries is deliberately generic, and the reason lives in
    `reason` for the log and the trace rather than in the text the caller reads.
    Telling a caller which pattern fired is a free oracle: it turns "refused"
    into a signal you can iterate against until you find the phrasing that gets
    through, which is exactly the loop a guardrail exists to deny.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__("This message was refused by a safety check")


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
