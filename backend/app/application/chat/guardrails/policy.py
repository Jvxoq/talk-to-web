"""The decision layer on top of the domain's detectors: what to do about
what they find.

`InputGuardPolicy` is deliberately the only place that turns a `Finding` into
an action. The detectors themselves never decide to block or redact - they
only recognize. That split is what lets `block_on_injection` flip from False
to True later without touching a single regex.
"""

from dataclasses import dataclass
from typing import Literal

from app.domain.chat.guardrails import Finding, detect_injection
from app.domain.chat.guardrails import redact_pii as redact_pii_text

GuardAction = Literal["allow", "redact", "block"]


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """What the policy decided about one piece of text.

    `text` is the input unchanged when `action` is "allow" or "block", and the
    redacted version when `action` is "redact" - a "block" verdict's `text` is
    never sent anywhere, so it is left as the original rather than something a
    caller might mistakenly forward.
    """

    action: GuardAction
    text: str
    findings: tuple[Finding, ...]

    def categories(self) -> tuple[str, ...]:
        """The distinct finding categories, in first-seen order - the shape a
        log line or a trace attribute wants, not a `Finding` object itself."""
        seen: list[str] = []
        for finding in self.findings:
            name = finding.category.value
            if name not in seen:
                seen.append(name)
        return tuple(seen)


class InputGuardPolicy:
    """Composes the domain detectors into one verdict for a user message.

    `block_on_injection` starts False in production. That is not an oversight
    - the injection patterns here are heuristic, and shipping them with
    blocking already on on day one means the first false positive is a user
    with a legitimately refused message and no way to know why. With blocking
    off, every injection finding is still attached to the verdict (so it
    reaches the log and the trace) and the action falls through to whatever
    the PII check decided; only once that data shows the false-positive rate
    is low does flipping the setting turn the same findings into a block.
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
