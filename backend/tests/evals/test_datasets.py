"""The eval datasets themselves, checked for free.

`python -m evals` spends real money on real APIs, and `evals.cases.load_cases`
validates eagerly precisely so a typo surfaces before the first paid call —
but only once someone runs it. These tests move that check into the
four-second suite, where a malformed JSONL line fails a PR instead of an
afternoon.

Nothing here calls a model, touches Qdrant or reads an API key: importing
`evals.cases` pulls in pydantic and nothing else, so this belongs in the fast
suite rather than under `tests/integration/`.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from app.adapters.llm.approximate_token_counter import ApproximateTokenCounter
from app.application.chat.models import ChatMessage
from app.settings import Settings
from evals.cases import BEHAVIORS, EvalCase, expanded_history, load_cases
from evals.recorded_web import INDEX_NAME, RecordedWeb, normalize_url

_EVALS_DIR = Path(__file__).parents[2] / "evals"
_FIXTURES_DIR = _EVALS_DIR / "fixtures"

_DATASETS = ["tools.jsonl", "rag.jsonl"]
"""The two files `evals.__main__._select_cases` loads, by the same names."""

_CI_TAG = "ci"
"""The tag the CI job selects with. See `.github/workflows/backend.yml`."""


def _load(name: str) -> list[EvalCase]:
    return load_cases(Path(name))


@pytest.mark.parametrize("name", _DATASETS)
def test_dataset_parses_and_is_not_empty(name: str) -> None:
    assert _load(name)


@pytest.mark.parametrize("name", _DATASETS)
def test_suite_matches_its_filename(name: str) -> None:
    """A case whose `suite` does not match the file it lives in is dropped
    silently: `_select_cases` filters on `case.suite == name` after loading
    the file named for that suite. It would not fail, it would just never
    run."""
    expected_suite = name.removesuffix(".jsonl")
    assert all(case.suite == expected_suite for case in _load(name))


@pytest.mark.parametrize("name", _DATASETS)
def test_named_fixtures_exist(name: str) -> None:
    """`EvalCase.fixtures` is documentation rather than a load instruction —
    `EvalHarness.index_fixtures` indexes the whole directory — so a stale
    filename here would never raise. It would only make a report explain a
    miss with a file that no longer exists."""
    missing = [
        fixture
        for case in _load(name)
        for fixture in case.fixtures
        if not (_FIXTURES_DIR / fixture).is_file()
    ]
    assert not missing


@pytest.mark.parametrize("name", _DATASETS)
def test_expected_sources_name_real_fixtures(name: str) -> None:
    """Unlike `fixtures`, these are load-bearing: `hit_rate_at_k` compares
    them against the source labels a reply actually returned, and a typo
    here fails every case that carries it for a reason that looks like a
    retrieval regression."""
    missing = [
        source
        for case in _load(name)
        for source in case.expect.expected_sources
        if not source.startswith("http") and not (_FIXTURES_DIR / source).is_file()
    ]
    assert not missing


@pytest.mark.parametrize("name", _DATASETS)
def test_every_case_checks_something(name: str) -> None:
    """A case with an empty `expect` costs an API call and proves nothing."""
    unchecked = [
        case.id
        for case in _load(name)
        if not (
            case.expect.tools
            or case.expect.expected_sources
            or case.expect.must_contain
            or case.expect.must_not_contain
        )
    ]
    assert not unchecked


@pytest.mark.parametrize("name", _DATASETS)
def test_ci_tagged_cases_exist(name: str) -> None:
    """The gate on the gate. CI runs `--tag ci`, and a selection that matches
    nothing is a warning and an exit code of 0 — a green job that tested
    nothing, which is the same failure mode the integration job guards
    against with its skip check."""
    assert any(_CI_TAG in case.tags for case in _load(name))


def test_the_empty_account_is_exercised() -> None:
    """`ToolRoutingPolicy` reads an empty account in the opposite direction to
    a full one: it refuses `retrieve_documents` and stops holding `search_web`
    back. A dataset with no `owner: "empty"` case tests one arm of a two-armed
    gate, and the untested arm is every account's first session."""
    cases = [case for name in _DATASETS for case in _load(name)]
    assert [case.id for case in cases if case.owner == "empty"]


def test_an_ordering_rule_is_exercised() -> None:
    """The one rule the routing gate exists for is an ordering rule -
    retrieval answers before a web search is paid for. Membership alone cannot
    state it: a run that searched first and retrieved afterwards has exactly
    the same `tools` set as a run that got it right."""
    cases = [case for name in _DATASETS for case in _load(name)]
    assert [case.id for case in cases if case.expect.tool_order]


@pytest.mark.parametrize("name", _DATASETS)
def test_every_task_has_one_behavior(name: str) -> None:
    """The report groups by behavior and `--tag b1` selects by it. A task with
    no behavior tag drops out of every group, and one with two is counted
    twice."""
    wrong = [
        case.id for case in _load(name) if len([tag for tag in case.tags if tag in BEHAVIORS]) != 1
    ]
    assert not wrong


def test_behavior_tags_match_the_design_table() -> None:
    """`eval-design.md` lists B1 to B6. A seventh tag here has no row there."""
    assert BEHAVIORS == ("b1", "b2", "b3", "b4", "b5", "b6")


_WEB_DIR = _FIXTURES_DIR / "web"
_URL = re.compile(r"https?://[^\s)>\]\"']+")


def test_recordings_load_and_every_file_in_web_is_indexed() -> None:
    """A recording nobody lists is dead weight, and a listed one that is
    missing would only fail on the first paid run. Loading checks the second;
    this checks the first."""
    RecordedWeb.load(_WEB_DIR)
    index = json.loads((_WEB_DIR / INDEX_NAME).read_text())
    listed = {*index["pages"].values(), *(entry["file"] for entry in index["searches"])}
    on_disk = {
        str(path.relative_to(_WEB_DIR))
        for path in _WEB_DIR.rglob("*")
        if path.is_file() and path.name != INDEX_NAME
    }

    assert on_disk == listed


def test_recorded_urls_use_the_reserved_example_tld() -> None:
    """`.example` can never resolve. A recording for a real host would look
    live, and the model could answer from memory of the real page."""
    index = json.loads((_WEB_DIR / INDEX_NAME).read_text())
    urls = [*index["pages"], *index["unreachable"]]

    assert [url for url in urls if not (urlsplit(url).hostname or "").endswith(".example")] == []


@pytest.mark.parametrize("name", _DATASETS)
def test_every_example_url_a_task_names_has_a_recording(name: str) -> None:
    """A task that pastes a `.example` URL with no recording would fetch
    nothing, and a model that cannot read the page fails for a reason that
    looks like a regression."""
    web = RecordedWeb.load(_WEB_DIR)
    index = json.loads((_WEB_DIR / INDEX_NAME).read_text())
    known = {normalize_url(url) for url in index["unreachable"]}
    missing = [
        (case.id, url)
        for case in _load(name)
        for url in _URL.findall(case.input)
        if (urlsplit(url).hostname or "").endswith(".example")
        and not web.page(url)
        and normalize_url(url) not in known
    ]
    assert not missing


@pytest.mark.parametrize("name", _DATASETS)
def test_web_tasks_in_ci_do_not_need_the_live_web(name: str) -> None:
    """`ci` rule 1: recorded pages make a B2 task offline, but a task that
    names a real host would still reach for the internet under `--web live`
    only. Keep every `ci` fetch task on a recorded URL, and every `ci` search
    task on a recorded search."""
    web = RecordedWeb.load(_WEB_DIR)
    offenders = [
        case.id
        for case in _load(name)
        if _CI_TAG in case.tags
        and (
            (
                "fetch_web_pages" in case.expect.tools
                and any(
                    not (urlsplit(url).hostname or "").endswith(".example")
                    for url in _URL.findall(case.input)
                )
            )
            or ("search_web" in case.expect.tools and web.search(case.input) is None)
        )
    ]
    assert not offenders


def test_fetch_task_needle_is_not_copied_from_the_input() -> None:
    """tools-017 once graded the URL the user pasted. A model that echoes the
    question satisfies that without using what it fetched."""
    echoed = [
        (case.id, needle)
        for name in _DATASETS
        for case in _load(name)
        for needle in case.expect.must_contain
        if needle.lower() in case.input.lower()
    ]
    assert echoed == []


_HISTORY_BUDGET = Settings.model_fields["agent_history_token_budget"].default
"""The production summarize threshold, read from the settings default so a
change to it fails here instead of quietly leaving a task under the line."""


def _history_tasks() -> list[EvalCase]:
    return [case for name in _DATASETS for case in _load(name) if case.history]


def test_a_multi_turn_task_is_exercised() -> None:
    """B5 is a multi-turn behavior. With no seeded task in the dataset, the
    schema field and the harness path that reads it run on nothing."""
    assert [case.id for case in _history_tasks() if "b5" in case.tags]


def test_summary_tasks_pass_the_summarize_threshold() -> None:
    """A task tagged `summary` claims the chat is long enough to be shortened.
    Under the threshold the agent never summarizes, and the task would pass
    without testing recall after a summary."""
    counter = ApproximateTokenCounter()
    short = [
        case.id
        for case in _history_tasks()
        if "summary" in case.tags
        and counter.count(
            [ChatMessage(role=role, content=text) for role, text in expanded_history(case.history)]
        )
        <= _HISTORY_BUDGET
    ]
    assert short == []


def test_summary_tag_needs_history() -> None:
    """The tag on a one-turn task would promise a summary no trial can reach."""
    summary_tasks = [case for name in _DATASETS for case in _load(name) if "summary" in case.tags]
    assert summary_tasks
    assert [case.id for case in summary_tasks if not case.history] == []


def test_history_needles_are_not_in_the_filler() -> None:
    """A needle that also appears in the generated filler would pass or fail
    for the filler's reason, not the model's."""
    clashes = [
        (case.id, needle)
        for case in _history_tasks()
        for needle in (*case.expect.must_contain, *case.expect.must_not_contain)
        for turn, (_, expanded) in zip(case.history, expanded_history(case.history), strict=True)
        if turn.pad_chars and needle.lower() in expanded.removeprefix(turn.text).lower()
    ]
    assert clashes == []
