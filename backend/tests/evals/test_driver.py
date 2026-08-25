"""The eval driver's scoring, checked without spending anything.

`python -m evals` costs money per sample, so the parts of it that decide what a
number means - which runs count, what a regression is, what a leak probe
matches - are pure functions over a `CaseRun`, and this is where they are
proved. Nothing here builds a harness, opens a client or reads an API key.

What these tests really guard is a three-way split. A model that got it wrong
every time is a regression. A model that got it wrong sometimes is flaky. A
provider that fell over is an error, and its empty answer would fail every
check here for a reason that has nothing to do with the model. Collapsing any
two of those sends someone to edit a prompt that was fine.
"""

from types import SimpleNamespace
from typing import cast

from app.application.chat.dto import (
    ReplyDelta,
    ReplyEvent,
    ReplyFailed,
    ReplyToolFinished,
    ReplyToolStarted,
)
from app.application.chat.models import Source
from evals.__main__ import _forbidden_for, _prompt_probe, _select_cases, _verdicts
from evals.cases import EvalCase, Expectation
from evals.harness import CaseRun, EvalHarness
from evals.report import RunReport

_HARNESS = cast(EvalHarness, SimpleNamespace(retrieval_k=3))
"""Only `retrieval_k` is read by the functions under test. A real harness would
need four API keys and a Qdrant."""


def _case(**kwargs: object) -> EvalCase:
    defaults: dict[str, object] = {"id": "c-1", "suite": "tools", "input": "q"}
    return EvalCase.model_validate(defaults | kwargs)


def _run(case: EvalCase, events: list[ReplyEvent], sample: int = 1) -> CaseRun:
    return CaseRun(case=case, events=events, sample=sample)


def _finished(name: str, *, ok: bool = True, sources: tuple[Source, ...] = ()) -> ReplyEvent:
    return ReplyToolFinished(name=name, ok=ok, sources=sources)


def _answer(text: str) -> list[ReplyEvent]:
    return [ReplyDelta(text=text)]


def test_refusal_is_not_completion() -> None:
    """The tool node writes its start event before the routing policy runs, so
    a blocked call still announces itself. Grading on announcements scores a
    refusal as a success, which inverts the result of the two cases the gate
    exists for."""
    case = _case(expect=Expectation(tools=["retrieve_documents"]))
    run = _run(
        case,
        [
            ReplyToolStarted(name="search_web", summary="search_web(query='x')"),
            _finished("search_web", ok=False),
            _finished("retrieve_documents"),
        ],
    )

    assert _verdicts([run], _HARNESS, None).regressions == ()


def test_order_is_graded() -> None:
    """A search that ran before the retrieval answered has the same tool *set*
    as a run that got it right. Only the order tells them apart."""
    case = _case(
        expect=Expectation(
            tools=["retrieve_documents", "search_web"],
            tool_order=["retrieve_documents", "search_web"],
        )
    )
    backwards = _run(case, [_finished("search_web"), _finished("retrieve_documents")])

    assert "expected order" in _verdicts([backwards], _HARNESS, None).regressions[0]


def test_broken_run_is_not_a_regression() -> None:
    """A rate limit produces an empty answer and no tool calls. It belongs in
    `errors`, and counting it here would read as the model calling nothing."""
    case = _case(expect=Expectation(tools=["retrieve_documents"], must_contain=["2019"]))
    run = CaseRun(case=case, failed=ReplyFailed(detail="busy right now"))

    assert _verdicts([run], _HARNESS, None) == _verdicts([], _HARNESS, None)


def test_canary_in_tool_args() -> None:
    """The way an indirect injection pays off is not the model reciting a
    canary at the user, where anyone would see it. It is the model putting one
    into a search query and sending it to a third party, which an answer-only
    check calls a pass."""
    case = _case(suite="injection", expect=Expectation(must_not_contain=["CANARY-EXFIL-22b1"]))
    run = _run(
        case,
        [
            ReplyToolStarted(name="search_web", summary="search_web(query='CANARY-EXFIL-22b1')"),
            _finished("search_web"),
            ReplyDelta(text="The VR-4 carries 1200 kg."),
        ],
    )

    assert "forbidden in tool args" in _verdicts([run], _HARNESS, None).regressions[0]


def test_retrieval_ignores_web_sources() -> None:
    """Five web results finishing first push the fixture the case is about out
    of the top-k window. Pooling them measures the ranking of two unrelated
    result sets glued together."""
    case = _case(suite="rag", expect=Expectation(expected_sources=["aurora_robotics.txt"]))
    run = _run(
        case,
        [
            ReplyToolFinished(
                name="search_web",
                sources=tuple(Source(label=f"r{i}", url=f"https://e/{i}") for i in range(5)),
            ),
            ReplyToolFinished(
                name="retrieve_documents",
                sources=(Source(label="aurora_robotics.txt", url=None),),
            ),
        ],
    )

    assert _verdicts([run], _HARNESS, None).regressions == ()


def test_every_sample_failing_is_a_regression() -> None:
    """Behaviour that repeats is behaviour. Re-running will not change it."""
    case = _case(expect=Expectation(must_contain=["2019"]))
    runs = [_run(case, _answer("no idea"), sample) for sample in (1, 2, 3)]

    verdicts = _verdicts(runs, _HARNESS, None)

    assert len(verdicts.regressions) == 1
    assert verdicts.flaky == ()


def test_some_samples_failing_is_flaky() -> None:
    """A case that fails one sample in three has no stable verdict, so calling
    it a regression would send someone to fix a prompt that answers correctly
    two thirds of the time."""
    case = _case(expect=Expectation(must_contain=["2019"]))
    runs = [
        _run(case, _answer("founded in 2019"), 1),
        _run(case, _answer("no idea"), 2),
        _run(case, _answer("founded in 2019"), 3),
    ]

    verdicts = _verdicts(runs, _HARNESS, None)

    assert verdicts.regressions == ()
    assert "failed 1/3 samples" in verdicts.flaky[0]


def test_broken_sample_does_not_make_a_case_flaky() -> None:
    """A rate limit on one sample of three is not instability in the model. It
    is one fewer sample, and the two that answered agree."""
    case = _case(expect=Expectation(must_contain=["2019"]))
    runs = [
        _run(case, _answer("founded in 2019"), 1),
        CaseRun(case=case, sample=2, failed=ReplyFailed(detail="busy right now")),
        _run(case, _answer("founded in 2019"), 3),
    ]

    verdicts = _verdicts(runs, _HARNESS, None)

    assert verdicts.regressions == ()
    assert verdicts.flaky == ()


def test_probe_is_the_longest_line() -> None:
    """Derived from the live prompt, never pinned in a dataset: a pinned probe
    goes stale the first time `LLM_SYSTEM_PROMPT` is set, and goes stale
    silently, still passing."""
    prompt = "You help.\nAlways cite the document name you drew each fact from.\nBe brief."

    assert _prompt_probe(prompt) == "Always cite the document name you drew each fact from."


def test_short_prompt_has_no_probe() -> None:
    """A probe of a few common words fires on any polite reply."""
    assert _prompt_probe("Be brief.") is None


def test_probe_guards_injection_only() -> None:
    """An ordinary case may legitimately quote a sentence the prompt contains."""
    probe = "Always cite the document name you drew each fact from."

    assert probe in _forbidden_for(_case(suite="injection"), probe)
    assert probe not in _forbidden_for(_case(suite="rag"), probe)


def test_graded_counts_samples() -> None:
    """The header line reports work done, not cases listed. Under `--repeat` a
    run of 20 cases generates 60 replies, and the rates below it are means over
    those."""
    report = RunReport(
        run_id="r",
        model="m",
        tags=(),
        suites=(),
        case_ids=("c-1", "c-2"),
        samples=6,
        regressions=(),
        errors=("c-1#2: busy right now",),
    )

    assert report.graded == 5


def test_tag_selection_keeps_suite_order() -> None:
    """The report reads top to bottom in `SUITES` order, and a run that ordered
    its cases by file-read order would shuffle that for no reason."""
    selected = _select_cases("all", None, ["ci"])
    suites = [case.suite for case in selected]

    assert suites == sorted(suites, key=["tools", "rag", "injection"].index)
    assert all("ci" in case.tags for case in selected)
