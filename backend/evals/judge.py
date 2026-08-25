"""An LLM judge scoring groundedness, relevance and correctness.

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

Three scores, not two, and the third is what fixed the second. Groundedness
and correctness are different questions, and the old shape asked one model
call to answer both against a single blob called "context" - which the caller
filled with the gold answer when it had one. Grading groundedness against the
gold answer means an invented fact that happens to match scores a perfect 1.0,
which is exactly the failure the metric exists to catch. Here `context` is the
source text the answer was supposed to be drawn from, `reference` is the
correct answer, and neither is allowed to stand in for the other.
"""

import json
import re
from dataclasses import dataclass

from loguru import logger

from app.application.chat.models import ChatMessage
from app.application.chat.ports import ChatModel, Tracer

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a strict evaluator grading one chat reply. Score each of these on "
    "a 0.0-1.0 scale:\n"
    "- groundedness: is every factual claim in the answer supported by the "
    "SOURCE TEXT below? Judge this against the source text ONLY. An answer "
    "that is true but not present in the source text is not grounded. An "
    "answer that correctly says the source text does not cover the question "
    "is fully grounded.\n"
    "- relevance: does the answer actually address the user's question?\n"
    "- correctness: does the answer agree with the REFERENCE ANSWER? Omit "
    "this score entirely if no reference answer is given.\n"
    "Respond with ONLY a JSON object, no prose, no markdown fence, in exactly "
    'this shape: {"groundedness": 0.0, "relevance": 0.0, "correctness": 0.0, '
    '"reason": "..."}'
)


@dataclass(frozen=True, slots=True)
class JudgeScore:
    groundedness: float
    relevance: float
    # `None` when the case supplied no reference answer, or when the judge
    # declined to score it. Distinct from 0.0, which means "contradicts the
    # reference" - averaging those together would punish every case that
    # simply has no gold answer.
    correctness: float | None
    reason: str


class Judge:
    """Scores one answer against its question, its sources and its gold answer."""

    def __init__(self, *, model: ChatModel, model_name: str, tracer: Tracer) -> None:
        self._model = model
        self._model_name = model_name
        self._tracer = tracer

    async def score(
        self, *, question: str, answer: str, context: str, reference: str | None = None
    ) -> JudgeScore | None:
        """Judge one answer. Never raises - a broken judge call costs one
        metric a `None`, never the eval run that asked for it."""
        if not answer.strip():
            # Nothing to grade. A `None` here reads correctly downstream: "not
            # judged", not "judged as zero".
            return None

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {question}\n\n"
                    f"SOURCE TEXT the answer had to be drawn from:\n{context or '(none)'}\n\n"
                    f"REFERENCE ANSWER:\n{reference or '(none given - omit correctness)'}\n\n"
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
                score = _parse(raw, has_reference=reference is not None)
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


def _parse(raw: str, *, has_reference: bool) -> JudgeScore | None:
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
            # Dropped unless the case actually gave a reference to compare
            # against. A model told to omit the field routinely emits it
            # anyway, and a score of "how well does this match nothing" would
            # go straight into the suite mean.
            correctness=_optional_correctness(payload) if has_reference else None,
            reason=str(payload.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _optional_correctness(payload: dict[str, object]) -> float | None:
    value = payload.get("correctness")
    if value is None or isinstance(value, bool):
        return None
    try:
        return _clamp(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
