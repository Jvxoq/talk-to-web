"""Golden-file guardrail tests against `evals/datasets/{redteam,benign}.jsonl`.

This is a pytest module, not part of the eval runner under `evals/`, precisely
because it is binary and deterministic: every case has one right answer
(a `GuardAction`), computed from regex matching alone, and needs zero API
calls to check. That is what lets it live in the four-second `quality` CI job
and gate every PR, unlike an LLM-judged eval that costs money and is noisy run
to run.

Each JSONL line records `expect.action_default` and `expect.action_strict`
rather than a single `expect.action`, because detection is config-independent
(the same regex either matches or it doesn't) but the action a `Finding` turns
into is not - `InputGuardPolicy.inspect` only blocks on an injection finding
when `block_on_injection=True`. Storing both lets one file exercise both
configurations named in the task: the shipped default
(`block_on_injection=False`) and the strict one that gates ever turning
blocking on.

`benign.jsonl` is the more important of the two files. `redteam.jsonl` proves
the detectors fire; `benign.jsonl` proves they do not fire on everything -
its false-positive rate under the strict configuration is the number that
decides whether `block_on_injection` can ever ship as `True`.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from app.application.chat.guardrails.policy import GuardAction, InputGuardPolicy

_DATASETS_DIR = Path(__file__).parent.parent.parent / "evals" / "datasets"

# Two configurations, matching the two the task calls out by name. Both keep
# `redact_pii=True`: PII handling is not what's being decided here, only
# whether an injection finding blocks.
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
        # Categories do not depend on the policy's action-selection - detection
        # runs unconditionally - so the strict config must find exactly the
        # same findings as the default one did.
        assert set(verdict.categories()) == set(case["expect"]["categories"])


# The worst benign false-positive rate this suite will tolerate, measured
# against `benign.jsonl`. Both are ceilings, not targets - lower them whenever a
# tightened pattern earns it, and never raise one to make a build pass.
#
# What is left at the default ceiling is one case: an opaque 40-character
# document id that matches the generic long-token secret pattern. That
# over-redaction is deliberate and documented in the detector - a missed API key
# costs far more than a redacted id. The strict ceiling adds one more: prose
# that quotes an attack while explaining it. No regex separates use from
# mention, which is one of the reasons `guardrail_block_on_injection` ships off.
_MAX_DEFAULT_FALSE_POSITIVE_RATE = 1 / 12
_MAX_STRICT_FALSE_POSITIVE_RATE = 2 / 12


class TestBenignFalsePositiveRate:
    """The number that gates ever setting `block_on_injection=True` in
    production: how often the detectors act on text that is not an attack.

    A "false positive" here is any action other than "allow" - a "redact" on
    benign text is not as costly as a wrongful "block", but it still rewrites
    a real user's message on a guess, so both count.
    """

    def test_reports_false_positive_rate_under_both_configurations(self) -> None:
        total = len(_BENIGN_CASES)
        assert total > 0

        default_fp = sum(1 for case in _BENIGN_CASES if case["expect"]["action_default"] != "allow")
        strict_fp = sum(1 for case in _BENIGN_CASES if case["expect"]["action_strict"] != "allow")

        default_rate = default_fp / total
        strict_rate = strict_fp / total

        # These are load-bearing on the *dataset*, not the detectors: they
        # prove the recorded expectations line up with what a false-positive
        # count over the file actually is, so the number in the module report
        # can be trusted. The dataset intentionally includes cases the
        # detectors get wrong on purpose - a document that quotes an attack,
        # and an opaque id long enough to look like a secret.
        #
        # A ceiling rather than an equality pin. This is a quality metric, so
        # the only direction worth failing a build over is upward: tightening a
        # pattern should not break the suite, and letting one drift should.
        assert default_rate <= _MAX_DEFAULT_FALSE_POSITIVE_RATE
        assert strict_rate <= _MAX_STRICT_FALSE_POSITIVE_RATE

    def test_every_benign_case_actually_matches_the_live_policy(self) -> None:
        """Belt and braces on top of the parametrized cases above: recompute
        the false-positive rate directly from `InputGuardPolicy.inspect`
        rather than from the recorded `expect` fields, so a stale dataset
        cannot make this class pass for the wrong reason."""
        default_fp = sum(
            1 for case in _BENIGN_CASES if _DEFAULT_POLICY.inspect(case["input"]).action != "allow"
        )
        strict_fp = sum(
            1 for case in _BENIGN_CASES if _STRICT_POLICY.inspect(case["input"]).action != "allow"
        )

        assert default_fp / len(_BENIGN_CASES) <= _MAX_DEFAULT_FALSE_POSITIVE_RATE
        assert strict_fp / len(_BENIGN_CASES) <= _MAX_STRICT_FALSE_POSITIVE_RATE


def test_dataset_actions_are_valid_guard_actions() -> None:
    """Cheap schema guard: a typo in a hand-edited JSONL line (`"blck"` for
    `"block"`) should fail loudly here, not surface as a silently-skipped
    assertion somewhere else."""
    valid_actions: set[GuardAction] = {"allow", "redact", "block"}
    for case in _ALL_CASES:
        assert case["expect"]["action_default"] in valid_actions
        assert case["expect"]["action_strict"] in valid_actions
