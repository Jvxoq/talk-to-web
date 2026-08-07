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

from app.application.chat.models import ToolCall


class ToolSpec(BaseModel):
    """A tool as the model sees it: a name, a reason to use it, and a schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    # JSON Schema. Produced by `model_json_schema()` rather than written by
    # hand, so the schema the model is shown can never drift from the model the
    # arguments are validated against.
    parameters: dict[str, Any]


class ToolOutcome(BaseModel):
    """
    What a tool produced, and whether it worked.

    Both halves are needed and they are not the same question. `content` always
    goes back to the model - even a failure is phrased as something the model can
    read and route around. `ok` is for the person watching: it is what turns the
    tool chip in the transcript red. Collapsing the two into a bare string, as an
    earlier draft did, made every failure silently indistinguishable from an
    answer.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    ok: bool = True


@runtime_checkable
class AgentTool(Protocol):
    """Structural: anything with a spec and a `run` is a tool."""

    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, arguments: Mapping[str, Any]) -> ToolOutcome: ...


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

    async def run(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        try:
            args = self.args_model.model_validate(dict(arguments))
        except ValidationError as error:
            # Returned rather than logged and swallowed: the model wrote these
            # arguments, so the model is the one who can fix them.
            logger.debug("Tool {} got invalid arguments: {}", self.name, error)
            return ToolOutcome(
                content=f"Invalid arguments for {self.name}: {error}", ok=False
            )

        try:
            return ToolOutcome(content=await self._run(args))  # type: ignore[arg-type]
        except Exception as error:
            logger.warning("Tool {} failed: {}", self.name, error)
            return ToolOutcome(content=f"{self.name} is unavailable right now.", ok=False)

    @abstractmethod
    async def _run(self, args: ArgsT) -> str:
        """Do the work. Arguments are already validated; exceptions are caught."""


class ToolRegistry:
    """
    Name to tool, and the only thing the graph knows about tools.

    The graph asks for `specs()` to tell the model what exists and calls
    `invoke()` to run what the model picked. Neither operation names a concrete
    tool, which is what makes the registry the single place a new capability is
    added.
    """

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Two tools share a name; the model could not tell them apart.")

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def invoke(self, call: ToolCall) -> ToolOutcome:
        """
        Run one requested call.

        Never raises, for any reason: an unknown name, bad arguments and a dead
        upstream all come back as a `ToolOutcome`. The response body is already
        streaming by the time this runs, so an exception here could only
        truncate a reply that the model was perfectly capable of finishing
        without that tool.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            # Models do occasionally invent a tool. Telling it which ones are
            # real costs one round trip; raising would cost the whole reply.
            available = ", ".join(self._tools) or "none"
            logger.warning("Model asked for unknown tool {!r}", call.name)
            return ToolOutcome(
                content=f"No tool named {call.name!r}. Available tools: {available}.",
                ok=False,
            )

        try:
            return await tool.run(call.arguments)
        except Exception as error:
            # `BaseTool.run` already catches, so reaching here means a tool that
            # does not extend it. Belt and braces, because one careless tool
            # must not be able to kill an open stream.
            logger.warning("Tool {} raised out of run(): {}", call.name, error)
            return ToolOutcome(content=f"{call.name} is unavailable right now.", ok=False)
