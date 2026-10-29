"""Inputs and outputs owned by the chat use cases — not the wire, not the database."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.application.chat.models import Source


class GenerateReplyInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    user_input: str
    owner_id: int
    temperature: float = 0.0
    # `None` means a one-off turn with no history.
    conversation_id: int | None = None


class ReplyDelta(BaseModel):
    """One piece of generated text."""

    model_config = ConfigDict(frozen=True)

    text: str


class ReplyToolStarted(BaseModel):
    """The agent started a tool. Shown so the wait has a reason."""

    model_config = ConfigDict(frozen=True)

    name: str
    summary: str


class ReplyToolFinished(BaseModel):
    """A tool returned. `ok=False` means the reply lost that grounding, not that it failed."""

    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool = True
    # Citations. Empty on a failed call.
    sources: tuple[Source, ...] = ()


class ReplySummarizing(BaseModel):
    """The agent is shortening a thread that outgrew its token budget.

    Not a tool event: the graph does it, not the model. Shown so the pause has
    a reason. `tokens_after` is `None` on the start event.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["start", "done"]
    tokens_before: int
    tokens_after: int | None = None


class ReplyFailed(BaseModel):
    """Generation broke after the stream had already opened."""

    model_config = ConfigDict(frozen=True)

    detail: str


class ReplyUsage(BaseModel):
    """Tokens this reply spent across every model call, condenser included.

    `elapsed_ms` is wall-clock time for the whole reply.
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: int
    model: str


class ReplyCompleted(BaseModel):
    """The model finished cleanly."""

    model_config = ConfigDict(frozen=True)


ReplyEvent = (
    ReplyDelta
    | ReplyToolStarted
    | ReplyToolFinished
    | ReplySummarizing
    | ReplyUsage
    | ReplyFailed
    | ReplyCompleted
)


class StartConversationInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    model_type: str
    owner_id: int


class RecordExchangeInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: int
    owner_id: int
    prompt_content: str
    response_content: str
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    is_success: bool = True
    status_code: int = 200
