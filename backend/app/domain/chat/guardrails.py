"""Guardrail detectors: PII redaction and prompt-injection recognition.

These are pattern-matching rules, not judgement calls, so they belong in the
domain the same way `is_blocked_address` does - a security policy expressed as
plain stdlib code, reviewable and unit-testable without a framework or a
network call.

Production runs this on a single worker with a hard concurrency limit and no
regex timeout available in the standard library, so every pattern here is
written to run in time linear in the input length: no nested quantifiers
`(a+)+`, no alternation repeated inside a repetition, no unbounded lookaround.
Catastrophic backtracking on attacker-controlled text is the threat model, not
raw length - length alone is handled by `max_scan_chars`, which every public
function in this module honours by scanning only a bounded prefix of its
input. A finding's `span` is therefore always within `[0, max_scan_chars)`.
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
    """One pattern match, kept for the log and the trace - never for display.

    `excerpt` is the matched text itself, so callers that redact must not put
    a `Finding` anywhere a user or the model can read it back; its only
    destinations are structured logging and the policy's own bookkeeping.
    """

    category: GuardCategory
    span: tuple[int, int]
    excerpt: str


def _bounded(text: str, max_scan_chars: int) -> str:
    """The prefix a detector is allowed to look at.

    Bounding at a hard character count, not a word or line boundary, is what
    makes the cost of every detector predictable regardless of how the input
    is shaped - a 200,000-char tool result costs exactly as much to scan as a
    `max_scan_chars`-long one.
    """
    if max_scan_chars <= 0:
        return ""
    return text[:max_scan_chars]


# ---------------------------------------------------------------------------
# PII detectors
# ---------------------------------------------------------------------------

# Ordinary `local@domain.tld` shape, not the full RFC 5322 grammar. A
# linear, conservative match beats a precise one that can be made slow.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Conservative on purpose: a run of digit groups separated by spaces or
# dashes, never dots, so a version string never matches. The digit-count
# filter below is what tells a phone number from a card number. The
# leading lookbehind is fixed-width and O(1), and keeps a dash-grouped
# identifier glued to a letter from matching at all.
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\+\d{1,3}[\s\-]?)?(?:\(\d{2,4}\)[\s\-]?)?\d{2,4}(?:[\s\-]\d{2,4}){2,4}"
)
_PHONE_DIGIT_RANGE = range(7, 16)  # E.164 tops out at 15 digits; 7 excludes short local codes.

# An ISO-8601 date is shape-wise a dash-grouped phone number, and ingested
# documents are full of dates. Without this check every one is redacted as
# a phone number, corrupting the content the user uploaded. Checked after
# the fact against a fixed-width literal shape, so it cannot backtrack.
_ISO_DATE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")


def _is_iso_date_like(candidate: str) -> bool:
    """Whether a phone-candidate string is shaped like a calendar date
    (`YYYY-MM-DD`) rather than a phone number, so it can be excluded from
    phone redaction. Only the shape and the month/day ranges are checked -
    it does not validate the date exists (e.g. Feb 30 still counts), which is
    fine: the goal is to recognise the *shape*, not to be a calendar."""
    if not _ISO_DATE_RE.match(candidate):
        return False
    _year, month, day = candidate.split("-")
    return 1 <= int(month) <= 12 and 1 <= int(day) <= 31


# Candidate card numbers: 13-19 digits, loosely grouped. The Luhn check
# below is what decides card against any long number.
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

# Each provider's key has a checkable prefix, so these are anchored
# literal patterns rather than a "looks random" guess that would
# false-positive on hashes, UUIDs and git SHAs.
_SECRET_PATTERNS: tuple[tuple[GuardCategory, re.Pattern[str]], ...] = (
    # OpenAI-shaped keys, still seen pasted from other tooling.
    (GuardCategory.PII_SECRET, re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    # Groq API keys - a former provider, still pasted from old configs.
    (GuardCategory.PII_SECRET, re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    # Together AI API keys - the provider this app calls directly.
    (GuardCategory.PII_SECRET, re.compile(r"\btgp_v1_[A-Za-z0-9_\-]{20,}\b")),
    # GitHub personal access tokens, classic and fine-grained.
    (GuardCategory.PII_SECRET, re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    (GuardCategory.PII_SECRET, re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    # AWS access key IDs: fixed 16 trailing uppercase-alnum chars after AKIA.
    (GuardCategory.PII_SECRET, re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # A generic bearer-token shape, the catch-all for a provider nobody wrote
    # a rule for. It also catches some hashes and session ids, which is the
    # trade-off: over-redacting an opaque token costs nothing, an unredacted
    # key in a trace costs a great deal. Benign eval case bn-008 is a known,
    # accepted instance of that over-redaction.
    (GuardCategory.PII_SECRET, re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")),
)


def _luhn_ok(digits: str) -> bool:
    """The Luhn checksum, which tells a real card number from a long number.

    Without it every 16-digit order or tracking number is flagged.
    """
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
    """Keep the earliest, then longest, match when spans overlap.

    The generic "long token" secret pattern can overlap a phone or email
    match on adversarial input; redaction must apply each span once, so
    overlaps are resolved before replacement rather than during it.
    """
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
    """Replace every detected PII span in the scanned prefix with a stable
    placeholder. Text beyond `max_scan_chars` is returned unscanned and
    unmodified - the caller decides, via the bound, how much of a message it
    is willing to pay to inspect."""
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

# "Ignore/disregard previous/above instructions" and close variants. Bounded
# alternation of fixed literal verbs, no repeated groups - linear.
_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:the\s+)?"
    r"(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)\b",
    re.IGNORECASE,
)
# "disregard the above" without the word "instructions" attached.
_OVERRIDE_ABOVE_RE = re.compile(r"\b(?:ignore|disregard)\s+the\s+above\b", re.IGNORECASE)

# "reveal/print/show your (system) prompt" - a request to exfiltrate the
# instructions rather than follow new ones.
_REVEAL_PROMPT_RE = re.compile(
    r"\b(?:reveal|print|show|repeat)\s+(?:your\s+|the\s+)?(?:system\s+)?prompt\b",
    re.IGNORECASE,
)

# "You are now <role>" - a role-reassignment attempt. Three shapes, in one
# bounded alternation, no repeated groups - linear:
#   - "you are now a/an/the <word>"       (a rogue assistant, an unfiltered AI)
#   - "you are now in <word> mode"        (developer mode, DAN mode)
#   - "you are now <ACRONYM>"             (DAN) - a bare word with no article
# The bare-word shape used to false-positive on quoted dialogue ("you are
# now trapped"), so it is captured separately and validated in
# `_role_reassign_valid`: accepted only when upper-cased like a persona.
_ROLE_REASSIGN_RE = re.compile(
    r"\byou\s+are\s+now\s+"
    r"(?:(?:a|an|the)\s+\w+"
    r"|in\s+\w+\s+mode\b"
    r"|(?P<acronym>[A-Za-z]{2,20})\b)",
    re.IGNORECASE,
)

# A line opening with "system:" or a markdown "### system" heading, faking
# a system message inside supplied text. Anchored to a line start. It can
# still fire mid-sentence, and the trade-off favours recall: the shape is
# rare in prose and common in payloads.
_FAKE_SYSTEM_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s*)?system\s*:", re.IGNORECASE | re.MULTILINE)

# Markdown image exfiltration: `![...](http(s)://...?...=<long value>)`.
# The long value after `=` is the signature of a URL smuggling captured
# text out to an attacker-controlled host. `[^\s)]*` cannot backtrack
# against itself.
_IMAGE_EXFIL_RE = re.compile(
    r"!\[[^\]]*\]\(https?://[^\s)]*\?[^\s)]*=[^\s)]{20,}\)",
    re.IGNORECASE,
)


def _accept_all(_match: re.Match[str]) -> bool:
    return True


def _role_reassign_valid(match: re.Match[str]) -> bool:
    """Filter for `_ROLE_REASSIGN_RE`'s bare-word branch (see the pattern's
    comment): a match is only kept when either an article/"in ... mode"
    branch fired (`acronym` is unset), or the bare word looks like a persona
    name rather than ordinary lower-case prose."""
    acronym = match.group("acronym")
    return acronym is None or acronym.isupper()


# Quoted attacks (benign eval case bn-004) match `_OVERRIDE_RE` exactly as
# live ones do. No regex separates a document *about* injection from one
# *containing* it. Left unfixed on purpose, and one of the reasons
# `guardrail_block_on_injection` defaults to False.
_InjectionPattern = tuple[GuardCategory, re.Pattern[str], Callable[[re.Match[str]], bool]]

_INJECTION_PATTERNS: tuple[_InjectionPattern, ...] = (
    (GuardCategory.INJECTION_OVERRIDE, _OVERRIDE_RE, _accept_all),
    (GuardCategory.INJECTION_OVERRIDE, _OVERRIDE_ABOVE_RE, _accept_all),
    (GuardCategory.INJECTION_EXFIL, _REVEAL_PROMPT_RE, _accept_all),
    (GuardCategory.INJECTION_ROLE, _ROLE_REASSIGN_RE, _role_reassign_valid),
    (GuardCategory.INJECTION_ROLE, _FAKE_SYSTEM_LINE_RE, _accept_all),
    (GuardCategory.INJECTION_EXFIL, _IMAGE_EXFIL_RE, _accept_all),
)

# The subset of the categories above that `strip_instructions` is allowed to
# act on. INJECTION_EXFIL is deliberately excluded even though it is an
# injection concern: an exfiltration image is content to redact as PII-shaped
# data (the URL), not an instruction *line* to blank out, and the reveal-prompt
# phrasing is often quoted verbatim in legitimate discussion of prompt
# injection - stripping is reserved for the override/role-reassignment moves
# that only ever appear as an attack.
_STRIPPABLE = frozenset({GuardCategory.INJECTION_OVERRIDE, GuardCategory.INJECTION_ROLE})


def detect_injection(text: str, *, max_scan_chars: int) -> tuple[Finding, ...]:
    """Every injection-shaped span in the scanned prefix. Detection never
    modifies the text - callers decide what to do with a finding, including
    doing nothing but logging it (see `InputGuardPolicy`)."""
    scanned = _bounded(text, max_scan_chars)
    findings = [
        Finding(category, match.span(), match.group())
        for category, pattern, is_valid in _INJECTION_PATTERNS
        for match in pattern.finditer(scanned)
        if is_valid(match)
    ]
    return tuple(sorted(findings, key=lambda f: f.span[0]))


def strip_instructions(text: str, *, max_scan_chars: int) -> tuple[str, tuple[Finding, ...]]:
    """Blank out only override/role-reassignment lines within the scanned
    prefix, replacing the matched span with `[instruction removed]`.

    This is not a general filter over the text: a page *about* prompt
    injection - explaining what "ignore previous instructions" attacks look
    like - is legitimate content for this app to summarize, so only the two
    categories that are never legitimate content to see verbatim inside a
    *tool result* are struck, and only within the bounded prefix.
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

# Common openings of a model declining to answer. Used to notice when a reply
# guardrail (or the model itself) has refused, so the caller can decide how to
# log or surface that separately from a normal answer. A short, fixed set of
# literal prefixes checked against the start of the (stripped, lowercased)
# text - no regex needed at all, so there is nothing here to backtrack.
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
    """Whether a reply opens the way a declined answer typically does.

    Deliberately a prefix check against a fixed phrase list, not a regex or a
    model call: refusals have a small, stable set of openings, and this only
    needs to be cheap and directionally right, not exhaustive.
    """
    head = text.strip().lower()[:64]
    return any(head.startswith(prefix) for prefix in _REFUSAL_PREFIXES)
