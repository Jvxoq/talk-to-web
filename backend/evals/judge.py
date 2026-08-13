"""An LLM judge scoring groundedness and answer relevance.

Lives here, in `evals/`, rather than in `app/`, so production carries no code
it never calls: nothing in the request path samples a live reply and asks a
second model to grade it. The day that changes - live sampling of production
traffic for quality monitoring - this class moves into
`app/application/chat/guardrails/`, gets wired into the composition root next
to `Condenser`, and is exercised by `tests/application/`.

Its posture is copied from `app.application.chat.agent.condenser.Condenser`
on purpose, not by convention: never raises, returns `None` on any failure
(a bad model response, a malformed reply, a dead provider), calls the model
with `tools=()` so a judge can never become a second agent, and answers
through the cheap `agent_condenser_model` rather than the model under test -
grading is not the thing being measured.
"""

import json
import re
from dataclasses import dataclass

from loguru import logger

from app.application.chat.models import ChatMessage
from app.application.chat.ports import ChatModel, Tracer

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a strict evaluator grading one chat reply. Score two things on a "
    "0.0-1.0 scale:\n"
    "- groundedness: does the answer only state things supported by the "
    "provided context, without inventing facts?\n"
    "- relevance: does the answer actually address the user's question?\n"
    "Respond with ONLY a JSON object, no prose, no markdown fence, in exactly "
    'this shape: {"groundedness": 0.0, "relevance": 0.0, "reason": "..."}'
)


@dataclass(frozen=True, slots=True)
class JudgeScore:
    groundedness: float
    relevance: float
    reason: str


class Judge:
    """Scores one answer against its question and context, or gives up quietly."""

    def __init__(self, *, model: ChatModel, model_name: str, tracer: Tracer) -> None:
        self._model = model
        self._model_name = model_name
        self._tracer = tracer

    async def score(self, *, question: str, answer: str, context: str) -> JudgeScore | None:
        """Judge one answer. Never raises - a broken judge call costs one
        metric a `None`, never the eval run that asked for it."""
        if not answer.strip():
            # Nothing to grade. A `None` here reads correctly downstream: "not
            # judged", not "judged as zero" - the same distinction `ReplyCost`
            # draws between unpriced and free.
            return None

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"Context the assistant had available:\n{context or '(none)'}\n\n"
                    f"Assistant's answer:\n{answer}"
                ),
            ),
        ]
        async with self._tracer.span(
            "eval.judge", kind="generation", model=self._model_name
        ) as span:
            try:
                parts: list[str] = []
                async for chunk in self._model.stream(
                    model=self._model_name,
                    temperature=0.0,
                    messages=messages,
                    # Deliberately empty, exactly like `Condenser._run`: a judge
                    # that calls tools is a second agent, not a grader.
                    tools=(),
                ):
                    if chunk.text:
                        parts.append(chunk.text)

                raw = "".join(parts).strip()
                score = _parse(raw)
                if score is None:
                    logger.warning(f"Judge produced unparseable output: {raw[:200]!r}")
                    span.record_error(ValueError("unparseable judge output"))
                    return None

                span.set(groundedness=score.groundedness, relevance=score.relevance)
                return score
            except Exception as error:
                span.record_error(error)
                logger.warning(f"Judge failed on {self._model_name}: {error}")
                return None


def _parse(raw: str) -> JudgeScore | None:
    """Pull a `JudgeScore` out of text that may be wrapped in prose or a
    markdown fence - models routinely do both even when told not to."""
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.removeprefix("json").strip()

    match = _JSON_OBJECT.search(candidate)
    if match is None:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    try:
        return JudgeScore(
            groundedness=_clamp(float(payload["groundedness"])),
            relevance=_clamp(float(payload["relevance"])),
            reason=str(payload.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
