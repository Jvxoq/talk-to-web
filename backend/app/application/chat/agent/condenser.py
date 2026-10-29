"""The one place text is shortened.

History summaries, tool-output condensing and document digests all go through
here. It calls `ChatModel` with no tools. Failure returns `None`, never raises.
"""

from collections.abc import Sequence

from loguru import logger

from app.application.chat.agent.usage import emit_usage
from app.application.chat.models import ChatMessage
from app.application.chat.ports import ChatModel, Tracer


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
        document_summary_prompt: str,
        tracer: Tracer,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self.max_chars = max_chars
        self._tool_condense_prompt = tool_condense_prompt
        self._summary_prompt = summary_prompt
        self._document_summary_prompt = document_summary_prompt
        self._tracer = tracer

    async def condense(self, text: str, *, focus: str) -> str | None:
        """Rewrite a tool result to keep only what answers `focus`."""
        if not text:
            return None
        # A safety ceiling on the condenser's own input.
        sliced = text[: self.max_chars]
        messages = [
            ChatMessage(role="system", content=self._tool_condense_prompt),
            ChatMessage(role="user", content=f"Focus: {focus}\n\n{sliced}"),
        ]
        return await self._run(messages, span_name="condense.tool_output")

    async def summarize(self, messages: Sequence[ChatMessage]) -> str | None:
        """Compress an older stretch of the thread into a short summary."""
        if not messages:
            return None
        payload = [ChatMessage(role="system", content=self._summary_prompt), *messages]
        return await self._run(payload, span_name="condense.summary")

    async def summarize_document(self, name: str, text: str) -> str | None:
        """Write a few sentences on what an uploaded document is about.

        Implements ingestion's `DocumentSummarizer`. Reads only the first
        `max_chars`, which is enough to say what a document is.
        """
        if not text.strip():
            return None
        messages = [
            ChatMessage(role="system", content=self._document_summary_prompt),
            ChatMessage(role="user", content=f"Filename: {name}\n\n{text[: self.max_chars]}"),
        ]
        return await self._run(messages, span_name="condense.document_summary")

    async def _run(self, messages: Sequence[ChatMessage], *, span_name: str) -> str | None:
        """One condenser turn. Returns `None` on any failure, never raises."""
        async with self._tracer.span(span_name, kind="generation", model=self._model_name) as span:
            try:
                parts: list[str] = []
                async for chunk in self._model.stream(
                    model=self._model_name,
                    temperature=0.0,
                    messages=messages,
                    # No tools: this must stay a cheap single call.
                    tools=(),
                ):
                    if chunk.text:
                        parts.append(chunk.text)
                    # Count condenser tokens too, or costly replies look cheap.

                    if chunk.usage is not None:
                        emit_usage(model=self._model_name, usage=chunk.usage)
                        span.set(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                return "".join(parts).strip() or None
            except Exception as error:  # noqa: BLE001
                span.record_error(error)
                logger.warning(f"Condenser failed on {self._model_name}: {error}")
                return None
