"""The conversation shapes the agent passes around.

These describe an LLM wire protocol - a turn, a tool request, a streamed step -
rather than a business rule, so they live in the application layer rather than
the domain. That placement is also what keeps `pydantic` out of `app.domain`,
which the import contract forbids.

Immutable by construction: a `ChatMessage` that has been appended to the graph
state is history, and history that can be edited in place is a bug waiting for
a checkpointer to replay it.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """One tool the model asked for, with the arguments it chose."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    # Deliberately untyped here. The arguments come from the model, so they are
    # unvalidated until the tool that owns them checks them against its own
    # schema - see `tools.base.BaseTool.run`.
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """One turn in the conversation, in the shape every provider agrees on."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # Set on an assistant turn that asked for tools.
    tool_calls: tuple[ToolCall, ...] = ()
    # Set on a tool turn, naming the call it answers. Providers reject a tool
    # result that cannot be paired back to its request.
    tool_call_id: str | None = None


class ModelChunk(BaseModel):
    """
    One step of a streaming model turn: text, or a decision to call tools.

    Both fields can be empty - a provider may emit bookkeeping chunks that carry
    neither - so consumers check rather than assume.
    """

    model_config = ConfigDict(frozen=True)

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class Source(BaseModel):
    """
    Where a piece of grounding came from, as the UI shows it - not as the
    model reads it.

    `url` is optional because not every source has one: a passage retrieved
    from the user's own upload is cited by the document it came from, which
    has no address to link to. A web result always sets it.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    url: str | None = None


class Passage(BaseModel):
    """
    One retrieved chunk of the user's own documents, with the file it came
    from.

    The chat context's own shape for this, kept apart from
    `app.domain.ingestion.value_objects.Chunk` on purpose: that type belongs to
    the ingestion context, and `EmbeddedKnowledgeRetriever` is the seam that
    translates one into the other, so chat never has to import ingestion's
    domain to know what a passage is.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    source: str


class SearchResult(BaseModel):
    """What a web search returned: the flattened text the model reads, and the
    sources behind it for the UI to cite."""

    model_config = ConfigDict(frozen=True)

    text: str
    sources: tuple[Source, ...] = ()
