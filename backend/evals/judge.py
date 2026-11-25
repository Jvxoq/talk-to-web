"""Three yes/no LLM judges: grounded, eligible and abstained.

Lives here, in `evals/`, rather than in `app/`, so production carries no code
it never calls. Its posture is copied from
`app.application.chat.agent.condenser.Condenser` on purpose: never raises,
returns `None` on any failure, calls the model with `tools=()` so a judge can
never become a second agent, and answers through the cheap
`agent_condenser_model` rather than the model under test.

Yes/no, not a 0.0 to 1.0 score (eval-design.md, item 3). A yes/no verdict can
be checked against a person's yes/no label for the same trial, which is how
each judge earns trust before it gates anything: `evals.labels` counts how
often the two agree. One question per call, so a disagreement points at one
prompt, not at a prompt that asked three things at once.

Each judge answers a question about the answer, not "did the task pass".
`abstained` says yes when the answer declines ("that is not in your file"),
whether or not declining was right. Whether it was right is the task's call,
made by code, so the judge's own accuracy can be measured on its own.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal, get_args

from loguru import logger

from app.application.chat.models import ChatMessage
from app.application.chat.ports import ChatModel, Tracer

JudgeName = Literal["grounded", "eligible", "abstained"]
JUDGES: tuple[JudgeName, ...] = get_args(JudgeName)

JUDGES_BY_BEHAVIOR: dict[str, tuple[JudgeName, ...]] = {
    # B1 faithfulness and eligibility.
    "b1": ("grounded", "eligible"),
    # B2 goal accuracy (grounded in the fetched page) and failed tool reported.
    "b2": ("grounded", "abstained"),
    # B4 abstention recall, over-abstention and no invented file fact.
    "b4": ("grounded", "abstained"),
    # B6 answer at the cap.
    "b6": ("eligible",),
}
"""Which judges read a trial, by the task's behavior tag (eval-design.md, item 3).

B3 and B5 are graded by code alone, so a task tagged with them, or with no
behavior, gets no judge call.
"""

QUESTIONS: dict[JudgeName, str] = {
    "grounded": (
        "Is every factual claim in the ANSWER supported by the SOURCE TEXT? "
        "Judge against the source text only. A claim that is true but not in "
        "the source text is not supported. An answer that makes no factual "
        "claim, or only says the source text does not cover the question, is "
        "grounded. If the source text is empty and the answer states facts, it "
        "is not grounded."
    ),
    "eligible": (
        "Does the ANSWER respond to what the user asked? Yes if it answers the "
        "question, or clearly says it could not find the answer. No if it is "
        "off-topic, only asks a question back, or is filler that never "
        "addresses the request."
    ),
    "abstained": (
        "Does the ANSWER decline to give the answer, saying the information "
        "was not found, is not in the user's files, or could not be fetched? "
        "Yes if it declines, even if it also offers something else. No if it "
        "gives an answer to the question. Do not judge whether declining was "
        "right."
    ),
}

_SYSTEM_PROMPT = (
    "You are a strict evaluator. You answer one yes/no question about one chat "
    "reply.\n\nQUESTION: {question}\n\n"
    "Respond with ONLY a JSON object, no prose, no markdown fence, in exactly "
    'this shape: {{"reason": "<one sentence>", "verdict": true}}. '
    "verdict is true for yes and false for no."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One judge's answer for one trial."""

    judge: JudgeName
    # True means "yes" to the judge's question. For `abstained`, yes means the
    # answer declined, which is neither good nor bad on its own.
    verdict: bool
    reason: str


class Judge:
    """Asks one yes/no question about an answer, its question and its sources."""

    def __init__(self, *, model: ChatModel, model_name: str, tracer: Tracer) -> None:
        self._model = model
        self._model_name = model_name
        self._tracer = tracer
        # Running totals across every call, read once at the end of a run to
        # price the judge. Only what the provider reports: a call whose stream
        # carries no usage adds nothing, so this can under-count but never
        # invent tokens.
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    async def verdict(
        self, judge: JudgeName, *, question: str, answer: str, sources: str
    ) -> JudgeVerdict | None:
        """One judge's verdict. Never raises: a broken call costs one verdict a
        `None`, never the eval run that asked for it."""
        if not answer.strip():
            # Nothing to grade. `None` reads as "not judged", not as "no".
            return None

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT.format(question=QUESTIONS[judge])),
            ChatMessage(
                role="user",
                content=(
                    f"USER QUESTION:\n{question}\n\n"
                    f"SOURCE TEXT the agent's tools returned:\n{sources or '(none)'}\n\n"
                    f"ANSWER:\n{answer}"
                ),
            ),
        ]
        async with self._tracer.span(
            f"eval.judge.{judge}", kind="generation", model=self._model_name
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
                    if chunk.usage is not None:
                        self._prompt_tokens += chunk.usage.prompt_tokens
                        self._completion_tokens += chunk.usage.completion_tokens

                raw = "".join(parts).strip()
                result = _parse(judge, raw)
                if result is None:
                    logger.warning(f"Judge {judge} produced unparseable output: {raw[:200]!r}")
                    span.record_error(ValueError("unparseable judge output"))
                    return None

                span.set(verdict=result.verdict)
                return result
            except Exception as error:
                span.record_error(error)
                logger.warning(f"Judge {judge} failed on {self._model_name}: {error}")
                return None


def _parse(judge: JudgeName, raw: str) -> JudgeVerdict | None:
    """Pull a verdict out of text that may be wrapped in prose or a markdown
    fence - models routinely do both even when told not to."""
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

    # Only a real JSON boolean counts. "true" as a string or 1 as a number is
    # a model that ignored the shape, and guessing what it meant would put a
    # made-up verdict into the agreement counts.
    value = payload.get("verdict")
    if not isinstance(value, bool):
        return None
    return JudgeVerdict(judge=judge, verdict=value, reason=str(payload.get("reason", "")))
