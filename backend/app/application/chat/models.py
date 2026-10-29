"""The message shapes the agent passes around.

An LLM wire format, not business rules, so they live here and keep pydantic
out of the domain. Frozen, because history must not change after the fact.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """One tool the model asked for, with the arguments it chose."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    # Untyped: the tool validates them in `BaseTool.run`.
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """One turn in the conversation, in the shape every provider agrees on."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # Set on an assistant turn that asked for tools.
    tool_calls: tuple[ToolCall, ...] = ()
    # Set on a tool turn. Providers reject a result without it.
    tool_call_id: str | None = None


class TokenUsage(BaseModel):
    """Tokens one model turn spent, as the provider reported them.

    Reported, not estimated, because it is about money. Optional on
    `ModelChunk`, since not every provider reports it.
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelChunk(BaseModel):
    """One step of a streaming model turn: text, tool calls, or neither."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    # Only on the final chunk, if reported. `None` means unknown, not zero.
    usage: TokenUsage | None = None


class Source(BaseModel):
    """A citation for the UI. Uploaded documents have no `url`. Web results always do."""

    model_config = ConfigDict(frozen=True)

    label: str
    url: str | None = None


class Passage(BaseModel):
    """One retrieved chunk of the user's documents, with its file name.

    Chat's own type, so chat never imports ingestion's `Chunk`.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    source: str


class SearchResult(BaseModel):
    """Web search results: text for the model, sources for the UI."""

    model_config = ConfigDict(frozen=True)

    text: str
    sources: tuple[Source, ...] = ()
