"""The one place text is shortened.

All three compression points - history summarization, tool-output compression
and the digest written for an uploaded document - go through this single
service, so there is one prompt style, one failure policy and one place to
tune. It is a plain application service: it calls the
`ChatModel` port with an empty tool list, so it never imports LangGraph or
LangChain and never becomes a second agent.

Failure is a fallback, never a crash. A condenser that cannot answer costs the
reply some grounding, not the reply itself - callers keep the uncondensed (or
truncated) text when this returns `None`.
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
        # A hard slice before the call, so a pathological page cannot blow the
        # condenser's own budget. This is a safety ceiling, not the tuning knob.
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
        """Write the few sentences that say what an uploaded document is about.

        Satisfies `app.application.ingestion.ports.DocumentSummarizer`
        structurally - ingestion declares that port and never learns this class
        exists. Returning `None` on failure is the same contract as the two
        methods above, and matters more here: a digest is an enhancement, and a
        condenser having a bad day must not cost the user their upload.

        The same `max_chars` ceiling applies. A digest is written from the
        opening of a document rather than the whole of it on purpose - the
        first pages are what say what a document *is*, and paying to read a
        200-page PDF in full to produce three sentences is not a trade worth
        making.
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
                    # Deliberately empty: a condenser that calls tools is a
                    # second agent, and the whole point of this model is to be
                    # cheap.
                    tools=(),
                ):
                    if chunk.text:
                        parts.append(chunk.text)
                    # The condenser spends real tokens too - a reply that called
                    # three tools and condensed twice paid for that, and not
                    # counting it here would understate exactly the replies
                    # that cost the most.
                    if chunk.usage is not None:
                        emit_usage(model=self._model_name, usage=chunk.usage)
                        span.set(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                return "".join(parts).strip() or None
            except Exception as error:
                span.record_error(error)
                logger.warning(f"Condenser failed on {self._model_name}: {error}")
                return None
