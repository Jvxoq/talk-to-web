"""Tool base classes.

A new tool is one class: an args model, `_run`, and a line in the composition
root. Schema, validation and error handling are inherited.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from app.application.chat.guardrails.tool_output import ToolOutputGuard
from app.application.chat.models import Source, ToolCall


class ToolContext(BaseModel):
    """Facts about this turn that the model cannot set.

    Kept apart from the model-written arguments, so the model cannot name
    another owner or bypass routing. The tool node fills it: four fields from
    the run config, `prior_tools` from the history.
    """

    model_config = ConfigDict(frozen=True)

    owner_id: int
    # Retrieval reads only this thread. `None` owns no documents.
    conversation_id: int | None = None
    # Decided from the request text, not the history, which summarizing can lose.
    document_scoped: bool = False
    # `True` means "unknown": a wasted call is cheaper than hiding real documents.
    has_documents: bool = True
    # Tools that answered since the last user message. See `tools_run_this_turn`.
    prior_tools: frozenset[str] = frozenset()


class ToolSpec(BaseModel):
    """A tool as the model sees it: a name, a reason to use it, and a schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    # Generated from the args model, so it cannot drift from validation.
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """What `_run` returns on success. Only `run()` sets `ok=False`."""

    model_config = ConfigDict(frozen=True)

    content: str
    sources: tuple[Source, ...] = ()


class ToolOutcome(BaseModel):
    """A tool's result. `content` goes to the model, even on failure.

    `ok` and `sources` are for the user's tool chip.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    ok: bool = True
    sources: tuple[Source, ...] = ()


class ToolRoutingPolicy(BaseModel):
    """Documents before the web, on a question about the user's own files.

    A separate object so the registry never names concrete tools. The names
    come from the composition root. An empty retrieval still opens the search.

    With documents, retrieval goes first. With none, retrieval is refused and
    search is not held back, or the model would deadlock.
    """

    model_config = ConfigDict(frozen=True)

    # The private, cheap lookup that must be tried first.
    document_tool: str
    # The tool that is held back until it has been.
    web_search_tool: str

    def redirect(self, call: ToolCall, context: ToolContext) -> ToolOutcome | None:
        """The refusal to answer this call with, or `None` to let it run."""
        if call.name == self.document_tool:
            if context.has_documents:
                return None
            # Nothing is indexed, so the call is a sure miss. Refuse it.
            return ToolOutcome(
                content=(
                    "This user has not uploaded any documents, so "
                    f"{self.document_tool} has nothing to search. Answer from what you "
                    f"already know, or use {self.web_search_tool} if the question needs "
                    "current or public information."
                ),
                ok=False,
            )

        if call.name != self.web_search_tool:
            return None
        if not context.document_scoped or self.document_tool in context.prior_tools:
            return None
        if not context.has_documents:
            # A user with no documents can still phrase a document question.
            # Holding search back too would loop between two refusals.
            return None
        return ToolOutcome(
            content=(
                f"This question is about the user's uploaded documents. Call "
                f"{self.document_tool} first; use {self.web_search_tool} only if that "
                "comes back with nothing relevant."
            ),
            ok=False,
        )


@runtime_checkable
class AgentTool(Protocol):
    """Structural: anything with a spec and a `run` is a tool."""

    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolOutcome: ...


class BaseTool[ArgsT: BaseModel](ABC):
    """Declare the args model once. Schema, validation and errors follow.

    `run` never raises, because a raise would kill an open stream. Failures
    come back as text the model can act on.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolOutcome:
        try:
            args = self.args_model.model_validate(dict(arguments))
        except ValidationError as error:
            # Returned to the model, which wrote the arguments and can fix them.
            logger.debug("Tool {} got invalid arguments: {}", self.name, error)
            return ToolOutcome(content=f"Invalid arguments for {self.name}: {error}", ok=False)

        try:
            result = await self._run(args, context)  # type: ignore[arg-type]
            return ToolOutcome(content=result.content, sources=result.sources)
        except Exception as error:  # noqa: BLE001
            logger.warning("Tool {} failed: {}", self.name, error)
            return ToolOutcome(content=f"{self.name} is unavailable right now.", ok=False)

    @abstractmethod
    async def _run(self, args: ArgsT, context: ToolContext) -> ToolResult:
        """Do the work. Arguments are validated and exceptions are caught.

        Context is always passed, because optional parameters get forgotten.
        """


class ToolRegistry:
    """Maps names to tools. The only thing the graph knows about tools.

    It never names a concrete tool, so adding one needs no edit here.
    """

    def __init__(
        self,
        tools: Sequence[AgentTool],
        *,
        guard: ToolOutputGuard,
        routing: ToolRoutingPolicy | None = None,
    ) -> None:
        # `guard` is required, because a guard that defaults to off is off in
        # production. `routing` may be `None`: no order is a valid setup when
        # there is no document tool.
        self._tools = {tool.spec.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Two tools share a name; the model could not tell them apart.")
        self._guard = guard
        if routing is not None:
            missing = {routing.document_tool, routing.web_search_tool} - set(self._tools)
            if missing:
                # Otherwise the rule would silently never fire.
                raise ValueError(f"Routing policy names unregistered tools: {sorted(missing)}")
        self._routing = routing

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolOutcome:
        """Run one call for the person in `context`.

        Never raises: the stream is already open. Every tool result passes
        through here, so the fence and the routing policy live here and
        nowhere else.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            # The model invented a tool. List the real ones. Not fenced: this
            # is our own text, not external content.
            available = ", ".join(self._tools) or "none"
            logger.warning("Model asked for unknown tool {!r}", call.name)
            return ToolOutcome(
                content=f"No tool named {call.name!r}. Available tools: {available}.",
                ok=False,
            )

        if self._routing is not None and (redirect := self._routing.redirect(call, context)):
            # Our own text, so not fenced. The refusal text says which rule fired.
            logger.info("Tool routing refused {}", call.name)
            return redirect

        try:
            outcome = await tool.run(call.arguments, context)
        except Exception as error:  # noqa: BLE001
            # Only a tool that skips `BaseTool` gets here. Our own text, not fenced.
            logger.warning("Tool {} raised out of run(): {}", call.name, error)
            return ToolOutcome(content=f"{call.name} is unavailable right now.", ok=False)

        if not outcome.ok:
            # Failure text is ours, not fetched content, so no fence.
            return outcome

        fenced_content, findings = self._guard.wrap(tool=call.name, content=outcome.content)
        if findings:
            # Findings are logged only. They never block a successful call.

            categories = ", ".join(sorted({finding.category.value for finding in findings}))
            logger.warning(
                "Tool {} output contained {} finding(s) ({}); stripped before returning",
                call.name,
                len(findings),
                categories,
            )

        return ToolOutcome(content=fenced_content, ok=outcome.ok, sources=outcome.sources)
