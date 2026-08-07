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
