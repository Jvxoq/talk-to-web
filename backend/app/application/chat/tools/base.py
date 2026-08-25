"""What a tool is, and the machinery every tool gets for free.

The point of this module is that adding a capability to the agent costs one
class: declare a Pydantic args model, implement `_run`, register it at the
composition root. JSON Schema, argument validation and failure handling are
inherited, so no tool hand-writes them and no two tools disagree about them.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from app.application.chat.guardrails.tool_output import ToolOutputGuard
from app.application.chat.models import Source, ToolCall


class ToolContext(BaseModel):
    """Who this turn belongs to, and what has already happened on it.

    Separate from the arguments on purpose: arguments are written by the model,
    and none of this is negotiable by it. A model that could name the owner it
    wanted to search would be a model that could read anyone's documents; a
    model that could declare its own question "not about the uploaded files"
    would be a model that could talk its way out of the routing policy below.

    Every field is written by the tool node - see `agent.nodes.make_tool_node`.
    `owner_id`, `document_scoped` and `has_documents` come off the run config,
    decided once per request by `GenerateReply`; `prior_tools` is read out of
    the history, because it is the one of the four that changes between laps.
    """

    model_config = ConfigDict(frozen=True)

    owner_id: int
    # Whether the user's question on this turn is about files they supplied.
    # Decided from the request text, not from the checkpointed history: a
    # summarized thread can lose the turn it is about, and a gate that reads
    # its own input out of a lossy record fails open.
    document_scoped: bool = False
    # Whether this owner has anything to retrieve at all. Read from the
    # database once per request, alongside the names the digest is built from,
    # so it costs no extra query.
    #
    # Defaults to `True`, and the default is the honest answer to "we do not
    # know": every gate below is an optimisation, and the cost of guessing
    # wrong in the permissive direction is one wasted call, while guessing
    # wrong in the strict direction is a user with documents being told they
    # have none. `GenerateReply` sets this to `False` only when a read that
    # actually succeeded came back empty - never when the read itself failed.
    has_documents: bool = True
    # Every tool that has already run and answered since the user's last
    # message - requested is not enough, for the reason `tools_run_this_turn`
    # gives.
    prior_tools: frozenset[str] = frozenset()


class ToolSpec(BaseModel):
    """A tool as the model sees it: a name, a reason to use it, and a schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    # JSON Schema. Produced by `model_json_schema()` rather than written by
    # hand, so the schema the model is shown can never drift from the model the
    # arguments are validated against.
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """
    What `BaseTool._run` hands back on success: the text the model reads, and
    the sources behind it for the person watching.

    Kept separate from `ToolOutcome` because a tool never decides `ok` -
    reaching `_run` at all already means the call succeeded from the tool's
    point of view. `ok=False` is what `run()` writes when `_run` never got
    that far.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    sources: tuple[Source, ...] = ()


class ToolOutcome(BaseModel):
    """
    What a tool produced, and whether it worked.

    Both halves are needed and they are not the same question. `content` always
    goes back to the model - even a failure is phrased as something the model can
    read and route around. `ok` is for the person watching: it is what turns the
    tool chip in the transcript red. Collapsing the two into a bare string, as an
    earlier draft did, made every failure silently indistinguishable from an
    answer.

    `sources` rides along for the same audience as `ok`: the model never reads
    it back, since the citation-worthy detail (a URL, a document name) is
    already inline in `content` where the model can quote it.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    ok: bool = True
    sources: tuple[Source, ...] = ()


class ToolRoutingPolicy(BaseModel):
    """Docs before the web, on a question that is about the user's own files.

    A policy object rather than a branch in `ToolRegistry.invoke`, because the
    registry is deliberately ignorant of which concrete tools exist - it is the
    one place a capability can be added without editing it. The tool *names*
    therefore arrive from the composition root, next to where those tools are
    listed, and the registry only knows that some ordering may apply.

    This constrains the model without taking the choice away: a redirect is
    only possible while the document tool has not run on this turn, and a
    retrieval that came back empty still opens the search.

    Two rules live here, and they are the same rule read from both ends. On an
    account that has uploads, the document tool goes first. On an account that
    has none, the document tool is not worth a network round trip at all - and
    the web must *not* be held back waiting for it, because a hold-back that
    waits on a call which can never run is a deadlock the model can only escape
    by exhausting its lap budget.
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
            # The prompt tells the model to try this lookup whenever a question
            # names an entity it does not know, on the reasoning that coming
            # back empty costs only one call. That reasoning holds right up
            # until the account has nothing indexed, at which point every such
            # call is a guaranteed miss - and not a free one: it is an
            # embedding request and a vector-store round trip before the empty
            # answer comes back. Answered here instead, from a fact the model
            # cannot argue with.
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
            # Reachable: `is_document_scoped` matches phrasing ("what does my
            # PDF say"), so a user with an empty library can ask a
            # document-scoped question. Holding the search back here would
            # bounce the model between a retrieval that is refused above and a
            # search that is refused here until the lap budget runs out.
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
    """
    Declare the args model once; schema, validation and errors follow.

    `run` never raises. A tool that raises would take down a response the user
    is already watching stream, to punish a mistake the model could have fixed
    itself. Both failure modes return a string instead, which lands back in the
    conversation as a tool result and gives the model a chance to correct course
    or to answer without that tool.
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
            # Returned rather than logged and swallowed: the model wrote these
            # arguments, so the model is the one who can fix them.
            logger.debug("Tool {} got invalid arguments: {}", self.name, error)
            return ToolOutcome(content=f"Invalid arguments for {self.name}: {error}", ok=False)

        try:
            result = await self._run(args, context)  # type: ignore[arg-type]
            return ToolOutcome(content=result.content, sources=result.sources)
        except Exception as error:
            logger.warning("Tool {} failed: {}", self.name, error)
            return ToolOutcome(content=f"{self.name} is unavailable right now.", ok=False)

    @abstractmethod
    async def _run(self, args: ArgsT, context: ToolContext) -> ToolResult:
        """Do the work. Arguments are already validated; exceptions are caught.

        Every tool is handed the context whether or not it reads one. Passing it
        only to the tools that currently need it would mean changing the
        contract again the first time a second one does, and an optional
        parameter is an invitation to forget it.
        """


class ToolRegistry:
    """
    Name to tool, and the only thing the graph knows about tools.

    The graph asks for `specs()` to tell the model what exists and calls
    `invoke()` to run what the model picked. Neither operation names a concrete
    tool, which is what makes the registry the single place a new capability is
    added.
    """

    def __init__(
        self,
        tools: Sequence[AgentTool],
        *,
        guard: ToolOutputGuard,
        routing: ToolRoutingPolicy | None = None,
    ) -> None:
        # `guard` is required, not `ToolOutputGuard | None = None`. This
        # module's own tools argue that "an optional parameter is an
        # invitation to forget it" (see `BaseTool._run`'s docstring) - the same
        # reasoning applies here with higher stakes: a guard that defaults to
        # off is a guard that is off in production, silently, the first time a
        # caller forgets to pass one. Forcing every call site to supply a
        # guard means the choice to fence tool output is made once, at the
        # composition root, and cannot be quietly skipped anywhere else.
        #
        # `routing` is the one collaborator here that genuinely defaults to
        # nothing, and it is not the same kind of thing as `guard`: a fence
        # that is off is a vulnerability, while an ordering rule that is off is
        # a registry whose tools have no order - which is exactly the case for
        # a deployment (or a test) that has no document tool to put first.
        self._tools = {tool.spec.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Two tools share a name; the model could not tell them apart.")
        self._guard = guard
        if routing is not None:
            missing = {routing.document_tool, routing.web_search_tool} - set(self._tools)
            if missing:
                # A policy naming a tool that was never registered would silently
                # never fire, which is the worst way for a routing rule to fail.
                raise ValueError(f"Routing policy names unregistered tools: {sorted(missing)}")
        self._routing = routing

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolOutcome:
        """
        Run one requested call, on behalf of the person in `context`.

        Never raises, for any reason: an unknown name, bad arguments and a dead
        upstream all come back as a `ToolOutcome`. The response body is already
        streaming by the time this runs, so an exception here could only
        truncate a reply that the model was perfectly capable of finishing
        without that tool.

        This is also the single choke point every tool result passes through
        on its way back to the model, which is why the fence is applied here
        and nowhere else - a future caller cannot add a new tool, or a new way
        to invoke one, that bypasses it. The routing policy is applied here for
        the same reason: an ordering rule enforced in one node, or in one tool,
        is an ordering rule the next caller forgets.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            # Models do occasionally invent a tool. Telling it which ones are
            # real costs one round trip; raising would cost the whole reply.
            #
            # This string is NOT fenced. Unlike a tool's `content`, it is text
            # this app wrote from a fixed template plus the model's own
            # requested name and this registry's own tool names - nothing an
            # external source shaped. Fencing it anyway would suggest to the
            # model that its own name choice was untrusted external data,
            # which it is not; the fence exists for content that crossed a
            # trust boundary, and this string never did.
            available = ", ".join(self._tools) or "none"
            logger.warning("Model asked for unknown tool {!r}", call.name)
            return ToolOutcome(
                content=f"No tool named {call.name!r}. Available tools: {available}.",
                ok=False,
            )

        if self._routing is not None and (redirect := self._routing.redirect(call, context)):
            # Not fenced, and not an error the user ever sees: this is our own
            # sentence, telling the model to spend its next lap on the cheap
            # private lookup it skipped. The tool is never run, so nothing
            # crossed a trust boundary here.
            # Deliberately not spelling out which rule fired: the registry does
            # not know what the policy's rules are, and a log line that
            # restated one would be the first thing to go stale when a rule is
            # added. The refusal text itself says why, and it is right there in
            # the trace next to this line.
            logger.info("Tool routing refused {}", call.name)
            return redirect

        try:
            outcome = await tool.run(call.arguments, context)
        except Exception as error:
            # `BaseTool.run` already catches, so reaching here means a tool that
            # does not extend it. Belt and braces, because one careless tool
            # must not be able to kill an open stream.
            #
            # Same reasoning as the unknown-tool message above: this string is
            # ours, not the failed tool's output, so it is not fenced either.
            logger.warning("Tool {} raised out of run(): {}", call.name, error)
            return ToolOutcome(content=f"{call.name} is unavailable right now.", ok=False)

        if not outcome.ok:
            # A `BaseTool.run` failure path (invalid arguments, a caught
            # exception) is also this app's own text, not the external
            # content a tool fetched - same reasoning, no fence.
            return outcome

        fenced_content, findings = self._guard.wrap(tool=call.name, content=outcome.content)
        if findings:
            # Findings are for observability, never control flow: `invoke`
            # keeps its guarantee of never raising and never blocking a
            # successful tool call on what the fence stripped. Logged at
            # warning so a stripped instruction line in fetched content shows
            # up in both the trace and the log, without slowing the model
            # down waiting on a decision nobody is making here.
            categories = ", ".join(sorted({finding.category.value for finding in findings}))
            logger.warning(
                "Tool {} output contained {} finding(s) ({}); stripped before returning",
                call.name,
                len(findings),
                categories,
            )

        return ToolOutcome(content=fenced_content, ok=outcome.ok, sources=outcome.sources)
