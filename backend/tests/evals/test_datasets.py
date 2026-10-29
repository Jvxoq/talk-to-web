"""The eval datasets, checked for free in the fast suite.

A malformed case fails here, before any paid eval run. No model, Qdrant or key.
"""

from pathlib import Path

import pytest

from app.adapters.web.aiohttp_scraper import AiohttpWebContentFetcher
from app.application.chat.guardrails.policy import InputGuardPolicy
from app.settings import Settings
from evals.cases import SECURITY_DATASETS_DIR, SECURITY_SUITES, EvalCase, load_cases, suite_path

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
    """A case whose `suite` does not match its file would silently never run."""
    expected_suite = name.removesuffix(".jsonl")
    assert all(case.suite == expected_suite for case in _load(name))


@pytest.mark.parametrize("name", _DATASETS)
def test_named_fixtures_exist(name: str) -> None:
    """Named fixtures must exist, or reports cite missing files."""
    missing = [
        fixture
        for case in _load(name)
        for fixture in case.fixtures
        if not (_FIXTURES_DIR / fixture).is_file()
    ]
    assert not missing


@pytest.mark.parametrize("name", _DATASETS)
def test_expected_sources_name_real_fixtures(name: str) -> None:
    """Expected sources must exist, or a typo looks like a retrieval regression."""
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
    unchecked = [case.id for case in _load(name) if not _checks_something(case)]
    assert not unchecked


def _checks_something(case: EvalCase) -> bool:
    return bool(
        case.expect.tools
        or case.expect.expected_sources
        or case.expect.must_contain
        or case.expect.must_not_contain
        or case.expect.judge
    )


@pytest.mark.parametrize("name", _DATASETS)
def test_ci_tagged_cases_exist(name: str) -> None:
    """The `ci` tag must select something, or CI passes having tested nothing."""
    assert any(_CI_TAG in case.tags for case in _load(name))


def test_the_empty_account_is_exercised() -> None:
    """The routing gate behaves differently with no documents, so test that arm too."""
    cases = [case for name in _DATASETS for case in _load(name)]
    assert [case.id for case in cases if case.owner == "empty"]


def test_an_ordering_rule_is_exercised() -> None:
    """The routing rule is about order, which a tool set alone cannot check."""
    cases = [case for name in _DATASETS for case in _load(name)]
    assert [case.id for case in cases if case.expect.tool_order]


@pytest.mark.parametrize("name", _DATASETS)
def test_forbidden_canaries_exist_in_a_fixture(name: str) -> None:
    """Every `CANARY-` string must be in the corpus, or forbidding it proves nothing."""
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


# --- The security datasets in `evals-security/` ---------------------------------
#
# Built by `evals-security/build.py`. Each page runs through the production
# extractor, and fails if the case's fact would not reach the model.

_PAGE_HOST = "pages.evals.test"
"""`.test` never resolves, so a page the harness forgot to serve fails at DNS."""

_MIN_ATTACKS = 60
"""With zero successes, the 95% upper bound is about 3/n. Below 60 attacks a
clean run still cannot show a rate under the 5% target in `SLO.md`."""

needs_security_datasets = pytest.mark.skipif(
    not SECURITY_DATASETS_DIR.is_dir(), reason="evals-security/ is outside this checkout"
)


def _security(suite: str) -> list[EvalCase]:
    return load_cases(suite_path(suite))


def _page_text(case: EvalCase) -> str:
    assert case.page is not None
    return AiohttpWebContentFetcher._parse(case.page.html, case.page.url)


@needs_security_datasets
@pytest.mark.parametrize("suite", SECURITY_SUITES)
def test_security_dataset_parses_into_its_own_suite(suite: str) -> None:
    cases = _security(suite)
    assert cases
    assert all(case.suite == suite for case in cases)
    assert all(_checks_something(case) for case in cases)


@needs_security_datasets
def test_every_injection_case_has_a_page_on_the_unresolvable_host() -> None:
    """A case with no page, or a page on a real host, would fetch the live web."""
    cases = _security("injection_pages")
    urls = [case.page.url for case in cases if case.page is not None]

    assert len(urls) == len(cases)
    assert len(set(urls)) == len(urls)
    assert all(url.startswith(f"https://{_PAGE_HOST}/") for url in urls)


@needs_security_datasets
def test_the_input_guard_leaves_every_page_url_intact() -> None:
    """No page URL may look like a secret (32+ word characters), or the guard redacts it."""
    guard = InputGuardPolicy(
        redact_pii=True,
        block_on_injection=False,
        max_scan_chars=Settings.model_fields["guardrail_max_scan_chars"].default,
    )
    rewritten = [
        case.id
        for case in _security("injection_pages")
        if case.page is not None and case.page.url not in guard.inspect(case.input).text
    ]
    assert not rewritten


@needs_security_datasets
def test_the_answer_survives_the_text_extractor() -> None:
    """The extractor must keep each case's fact, or the model gets blamed."""
    lost = [
        f"{case.id}: {needle}"
        for case in _security("injection_pages")
        for needle in case.expect.must_contain
        if needle.lower() not in _page_text(case).lower()
    ]
    assert not lost


@needs_security_datasets
def test_every_forbidden_value_is_planted_in_its_own_page() -> None:
    """Each canary or poisoned value must be planted in the raw HTML."""
    unplanted = [
        f"{case.id}: {needle}"
        for case in _security("injection_pages")
        for needle in case.expect.must_not_contain
        if case.page is not None and needle not in case.page.html
    ]
    assert not unplanted


@needs_security_datasets
def test_every_judged_instruction_is_planted_in_its_own_page() -> None:
    """The page must carry `attack.instruction`, which is what the judge sees."""
    missing = [
        case.id
        for case in _security("injection_pages")
        if case.attack is not None
        and case.attack.instruction is not None
        and case.page is not None
        and case.attack.instruction not in case.page.html
    ]
    assert not missing


@needs_security_datasets
def test_each_security_rate_has_enough_attacks_to_clear_its_target() -> None:
    injection = _security("injection_pages")
    harmful = [c for c in _security("jailbreak") if c.expect.judge == "harmful_compliance"]
    benign = [c for c in _security("jailbreak") if c.expect.judge == "over_refusal"]

    assert len(injection) >= _MIN_ATTACKS
    assert len(harmful) >= _MIN_ATTACKS
    # Benign cases stop "refuse everything" from scoring perfectly.
    assert benign


@needs_security_datasets
def test_every_jailbreak_case_is_split_by_method() -> None:
    """Every case needs `attack.method`, or it vanishes from the per-method rows."""
    cases = _security("jailbreak")
    assert all(case.attack is not None and case.attack.method for case in cases)
    assert all(case.expect.should_refuse is not None for case in cases)
