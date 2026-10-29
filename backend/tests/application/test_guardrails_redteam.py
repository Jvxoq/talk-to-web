"""Golden-file guardrail tests against `evals/datasets/{redteam,benign}.jsonl`.

Deterministic and free, so it runs in the fast CI job. Each case records the
action for the default config and for `block_on_injection=True`.
`benign.jsonl` matters most: its false-positive rate decides whether blocking
can ever be turned on.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from app.application.chat.guardrails.policy import GuardAction, InputGuardPolicy

_DATASETS_DIR = Path(__file__).parent.parent.parent / "evals" / "datasets"

# Default and strict. Both redact PII. Only injection blocking differs.
_DEFAULT_POLICY = InputGuardPolicy(redact_pii=True, block_on_injection=False, max_scan_chars=10_000)
_STRICT_POLICY = InputGuardPolicy(redact_pii=True, block_on_injection=True, max_scan_chars=10_000)


def _load(name: str) -> list[dict[str, Any]]:
    path = _DATASETS_DIR / name
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _case_id(case: dict[str, Any]) -> str:
    return str(case["id"])


_REDTEAM_CASES = _load("redteam.jsonl")
_BENIGN_CASES = _load("benign.jsonl")
_ALL_CASES = _REDTEAM_CASES + _BENIGN_CASES


class TestGuardrailGoldenCases:
    """Every case in both datasets, checked against both policy configurations."""

    @pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
    def test_default_policy_matches_recorded_expectation(self, case: dict[str, Any]) -> None:
        verdict = _DEFAULT_POLICY.inspect(case["input"])

        assert verdict.action == case["expect"]["action_default"]
        assert set(verdict.categories()) == set(case["expect"]["categories"])

    @pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
    def test_strict_policy_matches_recorded_expectation(self, case: dict[str, Any]) -> None:
        verdict = _STRICT_POLICY.inspect(case["input"])

        assert verdict.action == case["expect"]["action_strict"]
        # Detection does not depend on config, so findings match.
        assert set(verdict.categories()) == set(case["expect"]["categories"])


# Maximum benign false-positive rates. Lower them when you can. Never raise
# one to pass a build. Default allows one known case (a long id redacted as a
# secret). Strict adds prose that quotes an attack.
_MAX_DEFAULT_FALSE_POSITIVE_RATE = 1 / 12
_MAX_STRICT_FALSE_POSITIVE_RATE = 2 / 12


class TestBenignFalsePositiveRate:
    """How often the detectors act on benign text. Any action but "allow" counts."""

    def test_reports_false_positive_rate_under_both_configurations(self) -> None:
        total = len(_BENIGN_CASES)
        assert total > 0

        default_fp = sum(1 for case in _BENIGN_CASES if case["expect"]["action_default"] != "allow")
        strict_fp = sum(1 for case in _BENIGN_CASES if case["expect"]["action_strict"] != "allow")

        default_rate = default_fp / total
        strict_rate = strict_fp / total

        # A ceiling, not an exact pin: only an increase should fail the build.
        assert default_rate <= _MAX_DEFAULT_FALSE_POSITIVE_RATE
        assert strict_rate <= _MAX_STRICT_FALSE_POSITIVE_RATE

    def test_every_benign_case_actually_matches_the_live_policy(self) -> None:
        """Recompute the rate from the live policy, not the recorded expectations."""
        default_fp = sum(
            1 for case in _BENIGN_CASES if _DEFAULT_POLICY.inspect(case["input"]).action != "allow"
        )
        strict_fp = sum(
            1 for case in _BENIGN_CASES if _STRICT_POLICY.inspect(case["input"]).action != "allow"
        )

        assert default_fp / len(_BENIGN_CASES) <= _MAX_DEFAULT_FALSE_POSITIVE_RATE
        assert strict_fp / len(_BENIGN_CASES) <= _MAX_STRICT_FALSE_POSITIVE_RATE


def test_dataset_actions_are_valid_guard_actions() -> None:
    """Every recorded action must be valid, so a typo fails here."""
    valid_actions: set[GuardAction] = {"allow", "redact", "block"}
    for case in _ALL_CASES:
        assert case["expect"]["action_default"] in valid_actions
        assert case["expect"]["action_strict"] in valid_actions
