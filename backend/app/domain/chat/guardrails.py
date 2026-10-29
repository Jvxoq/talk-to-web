"""Guardrail detectors: PII redaction and prompt-injection recognition.

Plain stdlib rules, so they live in the domain and test without I/O.

Regex has no timeout and one worker serves every stream, so every pattern is
linear: no nested quantifiers, no repeated alternation, no unbounded
lookaround. Every public function scans only the first `max_scan_chars`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class GuardCategory(Enum):
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_CARD = "pii_card"
    PII_SECRET = "pii_secret"  # noqa: S105 - a category label, not a credential
    INJECTION_OVERRIDE = "injection_override"
    INJECTION_ROLE = "injection_role"
    INJECTION_EXFIL = "injection_exfil"


@dataclass(frozen=True, slots=True)
class Finding:
    """One match, for logs and traces only.

    `excerpt` holds the raw matched text, so never show it to the user or model.
    """

    category: GuardCategory
    span: tuple[int, int]
    excerpt: str


def _bounded(text: str, max_scan_chars: int) -> str:
    """The prefix a detector may scan. A fixed count keeps the cost predictable."""
    if max_scan_chars <= 0:
        return ""
    return text[:max_scan_chars]


# ---------------------------------------------------------------------------
# PII detectors
# ---------------------------------------------------------------------------

# Simple `local@domain.tld` shape, not full RFC 5322, to stay linear.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Digit groups split by spaces or dashes, never dots, so versions do not match.
# The digit count below separates phones from cards. The lookbehind skips
# identifiers glued to a letter.
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\+\d{1,3}[\s\-]?)?(?:\(\d{2,4}\)[\s\-]?)?\d{2,4}(?:[\s\-]\d{2,4}){2,4}"
)
_PHONE_DIGIT_RANGE = range(7, 16)  # E.164 tops out at 15 digits; 7 excludes short local codes.

# ISO dates look like dash-grouped phone numbers. This stops them being redacted.
_ISO_DATE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")


def _is_iso_date_like(candidate: str) -> bool:
    """Whether a phone candidate is shaped like `YYYY-MM-DD`.

    Checks shape and month/day ranges only, not that the date exists.
    """
    if not _ISO_DATE_RE.match(candidate):
        return False
    _year, month, day = candidate.split("-")
    return 1 <= int(month) <= 12 and 1 <= int(day) <= 31


# 13-19 digits, loosely grouped. The Luhn check decides.
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

# Anchored on each provider's known prefix, so hashes and UUIDs do not match.
_SECRET_PATTERNS: tuple[tuple[GuardCategory, re.Pattern[str]], ...] = (
    # OpenAI keys.
    (GuardCategory.PII_SECRET, re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    # Groq keys (former provider).
    (GuardCategory.PII_SECRET, re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    # Together AI keys (current provider).
    (GuardCategory.PII_SECRET, re.compile(r"\btgp_v1_[A-Za-z0-9_\-]{20,}\b")),
    # GitHub personal access tokens, classic and fine-grained.
    (GuardCategory.PII_SECRET, re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    (GuardCategory.PII_SECRET, re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    # AWS access key IDs.
    (GuardCategory.PII_SECRET, re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # Catch-all for long tokens. It over-redacts some hashes (eval bn-008),
    # which is cheaper than leaking a key into a trace.
    (GuardCategory.PII_SECRET, re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")),
)


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum, so order and tracking numbers are not flagged as cards."""
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_PLACEHOLDERS: dict[GuardCategory, str] = {
    GuardCategory.PII_EMAIL: "[redacted:email]",
    GuardCategory.PII_PHONE: "[redacted:phone]",
    GuardCategory.PII_CARD: "[redacted:card]",
    GuardCategory.PII_SECRET: "[redacted:secret]",
}


def _pii_findings(text: str) -> list[Finding]:
    findings: list[Finding] = []

    for match in _EMAIL_RE.finditer(text):
        findings.append(Finding(GuardCategory.PII_EMAIL, match.span(), match.group()))

    for match in _PHONE_RE.finditer(text):
        group = match.group()
        if _is_iso_date_like(group):
            continue
        digit_count = sum(1 for char in group if char.isdigit())
        if digit_count in _PHONE_DIGIT_RANGE:
            findings.append(Finding(GuardCategory.PII_PHONE, match.span(), match.group()))

    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ \-]", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            findings.append(Finding(GuardCategory.PII_CARD, match.span(), match.group()))

    for category, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding(category, match.span(), match.group()))

    return findings


def _drop_overlaps(findings: list[Finding]) -> list[Finding]:
    """Keep the earliest, then longest, match when spans overlap."""
    ordered = sorted(findings, key=lambda f: (f.span[0], -(f.span[1] - f.span[0])))
    kept: list[Finding] = []
    cursor = -1
    for finding in ordered:
        start, end = finding.span
        if start >= cursor:
            kept.append(finding)
            cursor = end
    return kept


def redact_pii(text: str, *, max_scan_chars: int) -> tuple[str, tuple[Finding, ...]]:
    """Replace PII in the scanned prefix with placeholders.

    Text past `max_scan_chars` is returned unchanged.
    """
    scanned = _bounded(text, max_scan_chars)
    findings = _drop_overlaps(_pii_findings(scanned))

    if not findings:
        return text, ()

    pieces: list[str] = []
    cursor = 0
    for finding in findings:
        start, end = finding.span
        pieces.append(scanned[cursor:start])
        pieces.append(_PLACEHOLDERS[finding.category])
        cursor = end
    pieces.append(text[cursor:])

    return "".join(pieces), tuple(findings)


# ---------------------------------------------------------------------------
# Injection detectors
# ---------------------------------------------------------------------------

# "Ignore previous instructions" and close variants.
_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:the\s+)?"
    r"(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)\b",
    re.IGNORECASE,
)
# "disregard the above" without the word "instructions" attached.
_OVERRIDE_ABOVE_RE = re.compile(r"\b(?:ignore|disregard)\s+the\s+above\b", re.IGNORECASE)

# "Reveal your system prompt": an attempt to leak the instructions.
_REVEAL_PROMPT_RE = re.compile(
    r"\b(?:reveal|print|show|repeat)\s+(?:your\s+|the\s+)?(?:system\s+)?prompt\b",
    re.IGNORECASE,
)

# "You are now <role>". Three shapes: "a/an/the <word>", "in <word> mode",
# and a bare word. The bare word only counts when upper-case like "DAN",
# so "you are now trapped" does not match.
_ROLE_REASSIGN_RE = re.compile(
    r"\byou\s+are\s+now\s+"
    r"(?:(?:a|an|the)\s+\w+"
    r"|in\s+\w+\s+mode\b"
    r"|(?P<acronym>[A-Za-z]{2,20})\b)",
    re.IGNORECASE,
)

# A line starting "system:" or "### system", faking a system message.
_FAKE_SYSTEM_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?system\s*:", re.IGNORECASE | re.MULTILINE)

# Markdown image whose URL carries a long query value: data being smuggled out.
_IMAGE_EXFIL_RE = re.compile(
    r"!\[[^\]]*\]\(https?://[^\s)]*\?[^\s)]*=[^\s)]{20,}\)",
    re.IGNORECASE,
)


def _accept_all(_match: re.Match[str]) -> bool:
    return True


def _role_reassign_valid(match: re.Match[str]) -> bool:
    """Keep a bare-word match only when the word is upper-case."""
    acronym = match.group("acronym")
    return acronym is None or acronym.isupper()


# Quoted attacks (eval bn-004) match like real ones. No regex can tell them
# apart, which is one reason `guardrail_block_on_injection` defaults to False.
_InjectionPattern = tuple[GuardCategory, re.Pattern[str], Callable[[re.Match[str]], bool]]

_INJECTION_PATTERNS: tuple[_InjectionPattern, ...] = (
    (GuardCategory.INJECTION_OVERRIDE, _OVERRIDE_RE, _accept_all),
    (GuardCategory.INJECTION_OVERRIDE, _OVERRIDE_ABOVE_RE, _accept_all),
    (GuardCategory.INJECTION_EXFIL, _REVEAL_PROMPT_RE, _accept_all),
    (GuardCategory.INJECTION_ROLE, _ROLE_REASSIGN_RE, _role_reassign_valid),
    (GuardCategory.INJECTION_ROLE, _FAKE_SYSTEM_LINE_RE, _accept_all),
    (GuardCategory.INJECTION_EXFIL, _IMAGE_EXFIL_RE, _accept_all),
)

# Only override and role moves are stripped. Exfil phrasing often appears in
# legitimate writing about injection, so it is detected but kept.
_STRIPPABLE = frozenset({GuardCategory.INJECTION_OVERRIDE, GuardCategory.INJECTION_ROLE})


def detect_injection(text: str, *, max_scan_chars: int) -> tuple[Finding, ...]:
    """Injection-shaped spans in the scanned prefix. Never changes the text."""
    scanned = _bounded(text, max_scan_chars)
    findings = [
        Finding(category, match.span(), match.group())
        for category, pattern, is_valid in _INJECTION_PATTERNS
        for match in pattern.finditer(scanned)
        if is_valid(match)
    ]
    return tuple(sorted(findings, key=lambda f: f.span[0]))


def strip_instructions(text: str, *, max_scan_chars: int) -> tuple[str, tuple[Finding, ...]]:
    """Replace override and role matches in the scanned prefix with `[instruction removed]`.

    Not a general filter: a page about prompt injection is still valid content.
    """
    scanned = _bounded(text, max_scan_chars)
    all_findings = detect_injection(scanned, max_scan_chars=max_scan_chars)
    strippable = _drop_overlaps([f for f in all_findings if f.category in _STRIPPABLE])

    if not strippable:
        return text, tuple(all_findings)

    pieces: list[str] = []
    cursor = 0
    for finding in strippable:
        start, end = finding.span
        pieces.append(scanned[cursor:start])
        pieces.append("[instruction removed]")
        cursor = end
    pieces.append(text[cursor:])

    return "".join(pieces), tuple(all_findings)


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

# Common openings of a refusal, matched as plain prefixes.
_REFUSAL_PREFIXES = (
    "i can't help with that",
    "i cannot help with that",
    "i can't assist with that",
    "i cannot assist with that",
    "i'm sorry, but i can't",
    "i am sorry, but i cannot",
    "i won't help with that",
    "i will not help with that",
    "sorry, i can't",
    "sorry, i cannot",
    "as an ai, i cannot",
    "as an ai language model, i cannot",
)


def looks_like_refusal(text: str) -> bool:
    """Whether a reply opens like a refusal. Cheap and approximate on purpose."""

    head = text.strip().lower()[:64]
    return any(head.startswith(prefix) for prefix in _REFUSAL_PREFIXES)
