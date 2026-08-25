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
    # What the answer can be cited to - a document name, a page title and URL.
    # Empty on a failed call: there is nothing to cite when the tool found
    # nothing.
    sources: tuple[Source, ...] = ()


class ReplySummarizing(BaseModel):
    """The agent is shortening a thread that outgrew its token budget.

    Its own event rather than a tool event, because it is not something the
    model asked for: the graph does it on the user's behalf, between turns. The
    client is told so the pause has a reason on screen - a summarization is a
    whole model call during which no text arrives.

    `status` is `"start"` or `"done"`. `tokens_after` is only known once the
    shortened history exists, so it is `None` on the start event.
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
    """What this reply spent, totalled across every model call it made.

    Every lap counts, not just the answering one: a reply that called three
    tools and condensed twice paid for the condenser too, and a number that
    omitted that would understate the expensive replies by the most.

    `elapsed_ms` is wall-clock time for the whole reply - from the first byte
    of work to the last event - not just one model call.
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
