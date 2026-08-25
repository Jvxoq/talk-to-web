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

from pathlib import Path

import pytest

from evals.cases import EvalCase, load_cases

_EVALS_DIR = Path(__file__).parents[2] / "evals"
_FIXTURES_DIR = _EVALS_DIR / "fixtures"

_DATASETS = ["tools.jsonl", "rag.jsonl", "injection.jsonl"]
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
def test_forbidden_canaries_exist_in_a_fixture(name: str) -> None:
    """A canary that is nowhere in the corpus cannot leak, so a case forbidding
    it passes without the agent having resisted anything. Only `CANARY-`
    strings are checked: the other `must_not_contain` entries are ordinary
    facts from a *different* fixture, and those are meant to be absent from
    the one the case reads."""
    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in _FIXTURES_DIR.iterdir() if path.is_file()
    )
    unplanted = [
        f"{case.id}: {needle}"
        for case in _load(name)
        for needle in case.expect.must_not_contain
        if needle.startswith("CANARY-") and needle not in corpus
    ]
    assert not unplanted
