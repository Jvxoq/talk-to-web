"""Inputs and outputs owned by the chat use cases — not the wire, not the database."""

from pydantic import BaseModel, ConfigDict


class GenerateReplyInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    user_input: str
    temperature: float = 0.0
    # The agent's memory key. `None` means a one-off turn with no history, which
    # is what a client that never opened a conversation gets.
    conversation_id: int | None = None


class ReplyDelta(BaseModel):
    """One piece of generated text."""

    model_config = ConfigDict(frozen=True)

    text: str


class ReplyToolStarted(BaseModel):
    """The agent reached for a tool. Reported so the wait has a reason on screen."""

    model_config = ConfigDict(frozen=True)

    name: str
    summary: str


class ReplyToolFinished(BaseModel):
    """A tool came back. `ok=False` means the answer lost that grounding, not that it failed."""

    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool = True


class ReplyFailed(BaseModel):
    """Generation broke after the stream had already opened."""

    model_config = ConfigDict(frozen=True)

    detail: str


class ReplyCompleted(BaseModel):
    """The model finished cleanly."""

    model_config = ConfigDict(frozen=True)


ReplyEvent = ReplyDelta | ReplyToolStarted | ReplyToolFinished | ReplyFailed | ReplyCompleted


class StartConversationInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    model_type: str


class RecordExchangeInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: int
    prompt_content: str
    response_content: str
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    is_success: bool = True
    status_code: int = 200
