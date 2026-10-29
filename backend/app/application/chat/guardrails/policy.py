"""Decides what to do about guardrail findings.

Detectors only recognize. This is the only place that blocks or redacts.
"""

from dataclasses import dataclass
from typing import Literal

from app.domain.chat.guardrails import Finding, detect_injection
from app.domain.chat.guardrails import redact_pii as redact_pii_text

GuardAction = Literal["allow", "redact", "block"]


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """The policy's decision. `text` is redacted only when `action` is "redact"."""

    action: GuardAction
    text: str
    findings: tuple[Finding, ...]

    def categories(self) -> tuple[str, ...]:
        """Distinct finding categories in first-seen order, for logs and traces."""
        seen: list[str] = []
        for finding in self.findings:
            name = finding.category.value
            if name not in seen:
                seen.append(name)
        return tuple(seen)


class InputGuardPolicy:
    """Combines the detectors into one verdict for a user message.

    `block_on_injection` is off in production because the patterns are
    heuristic. Findings are still logged, to measure false positives first.
    """

    def __init__(self, *, redact_pii: bool, block_on_injection: bool, max_scan_chars: int) -> None:
        self._redact_pii = redact_pii
        self._block_on_injection = block_on_injection
        self._max_scan_chars = max_scan_chars

    def inspect(self, text: str) -> GuardVerdict:
        pii_findings: tuple[Finding, ...] = ()
        working_text = text

        if self._redact_pii:
            working_text, pii_findings = redact_pii_text(text, max_scan_chars=self._max_scan_chars)

        injection_findings = detect_injection(text, max_scan_chars=self._max_scan_chars)
        all_findings = pii_findings + injection_findings

        if injection_findings and self._block_on_injection:
            return GuardVerdict(action="block", text=text, findings=all_findings)

        if pii_findings:
            return GuardVerdict(action="redact", text=working_text, findings=all_findings)

        return GuardVerdict(action="allow", text=text, findings=all_findings)
