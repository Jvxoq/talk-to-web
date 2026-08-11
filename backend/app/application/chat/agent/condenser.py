"""The one place a conversation is shortened.

Both compression points - history summarization and tool-output compression -
go through this single service, so there is one prompt style, one failure
policy and one place to tune. It is a plain application service: it calls the
`ChatModel` port with an empty tool list, so it never imports LangGraph or
LangChain and never becomes a second agent.

Failure is a fallback, never a crash. A condenser that cannot answer costs the
reply some grounding, not the reply itself - callers keep the uncondensed (or
truncated) text when this returns `None`.
"""

from collections.abc import Sequence

from loguru import logger

from app.application.chat.models import ChatMessage
from app.application.chat.ports import ChatModel


class Condenser:
    """Summarizes history and condenses tool output through one cheap model."""

    def __init__(
        self,
        *,
        model: ChatModel,
        model_name: str,
        max_chars: int,
        tool_condense_prompt: str,
        summary_prompt: str,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self.max_chars = max_chars
        self._tool_condense_prompt = tool_condense_prompt
        self._summary_prompt = summary_prompt

    async def condense(self, text: str, *, focus: str) -> str | None:
        """Rewrite a tool result to keep only what answers `focus`."""
        if not text:
            return None
        # A hard slice before the call, so a pathological page cannot blow the
        # condenser's own budget. This is a safety ceiling, not the tuning knob.
        sliced = text[: self.max_chars]
        messages = [
            ChatMessage(role="system", content=self._tool_condense_prompt),
            ChatMessage(role="user", content=f"Focus: {focus}\n\n{sliced}"),
        ]
        return await self._run(messages)

    async def summarize(self, messages: Sequence[ChatMessage]) -> str | None:
        """Compress an older stretch of the thread into a short summary."""
        if not messages:
            return None
        payload = [ChatMessage(role="system", content=self._summary_prompt), *messages]
        return await self._run(payload)

    async def _run(self, messages: Sequence[ChatMessage]) -> str | None:
        """One condenser turn. Returns `None` on any failure, never raises."""
        try:
            parts: list[str] = []
            async for chunk in self._model.stream(
                model=self._model_name,
                temperature=0.0,
                messages=messages,
                # Deliberately empty: a condenser that calls tools is a second
                # agent, and the whole point of this model is to be cheap.
                tools=(),
            ):
                if chunk.text:
                    parts.append(chunk.text)
            return "".join(parts).strip() or None
        except Exception as error:
            logger.warning(f"Condenser failed on {self._model_name}: {error}")
            return None
