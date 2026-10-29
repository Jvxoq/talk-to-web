"""Whether a question is about the user's own uploaded files.

Plain stdlib rules, like `guardrails.py`. `ToolRegistry.invoke` acts on the
answer. Patterns stay linear and scan only `max_scan_chars`, because regex
cannot time out on the single worker.
"""

import re
from collections.abc import Sequence

# A question shows its subject well before this. Keeps scan cost flat.
MAX_SCAN_CHARS = 4_000

# Nouns for things a user hands over.
_DOC_NOUN = r"(?:documents?|docs?|files?|pdfs?|uploads?|attachments?)"

# Nouns that count only with "uploaded" or "attached". "The report" may be public.
_QUALIFIED_NOUN = r"(?:report|contract|paper|memo|notes?|deck|slides?|spreadsheet|transcript)s?"

_HANDED_OVER = r"(?:uploaded|attached|shared|sent|provided|gave)"

# Separate patterns, so a wrong match traces back to one rule.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "my documents", "our uploaded files", "my own docs"
    re.compile(rf"\b(?:my|our)\s+(?:own\s+|uploaded\s+|attached\s+)?{_DOC_NOUN}\b", re.IGNORECASE),
    # "my uploaded report", "our attached contract"
    re.compile(
        rf"\b(?:my|our)\s+(?:uploaded|attached)\s+{_QUALIFIED_NOUN}\b",
        re.IGNORECASE,
    ),
    # "the PDF", "the attachment", "the doc"
    re.compile(rf"\bthe\s+{_DOC_NOUN}\b", re.IGNORECASE),
    # "the uploaded report", "the attached memo"
    re.compile(rf"\bthe\s+(?:uploaded|attached)\s+{_QUALIFIED_NOUN}\b", re.IGNORECASE),
    # "the file I uploaded", "documents I sent you", "the report I gave you"
    re.compile(
        rf"\b(?:{_DOC_NOUN}|{_QUALIFIED_NOUN})\s+(?:that\s+|which\s+)?I\s+{_HANDED_OVER}\b",
        re.IGNORECASE,
    ),
    # "I uploaded a spec last week - what does it say about X?"
    re.compile(r"\bI\s+(?:uploaded|attached)\b", re.IGNORECASE),
    # "uploaded files", "attached pdf" - without a possessive in front.
    re.compile(
        rf"\b(?:uploaded|attached)\s+(?:{_DOC_NOUN}|{_QUALIFIED_NOUN})\b",
        re.IGNORECASE,
    ),
)


# Shorter names collide with ordinary words, like "a.pdf" or "no.txt".
_MIN_NAME_CHARS = 4


def _names_pattern(document_names: Sequence[str]) -> re.Pattern[str] | None:
    """One pattern matching any document by full name or stem, or `None`.

    Filenames are user input, so they are always escaped.
    """
    forms: dict[str, None] = {}
    for name in document_names:
        stem = name.rpartition(".")[0] or name
        for candidate in (name, stem):
            if len(candidate) >= _MIN_NAME_CHARS:
                forms.setdefault(candidate, None)
    if not forms:
        return None
    # Longest first, so the most specific name wins. Literals only, so linear.
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])", re.IGNORECASE)


def is_document_scoped(
    text: str,
    *,
    document_names: Sequence[str] = (),
    max_scan_chars: int = MAX_SCAN_CHARS,
) -> bool:
    """Whether the user is asking about files they supplied.

    Conservative on purpose: a miss costs one web search, a false match costs
    a redirect. `document_names` comes from the database and catches questions
    that name an owned file. It matches names, not subjects. The digest in
    `GenerateReply` covers subjects.
    """

    if max_scan_chars <= 0:
        return False
    bounded = text[:max_scan_chars]
    if any(pattern.search(bounded) for pattern in _PATTERNS):
        return True
    names = _names_pattern(document_names)
    return names is not None and names.search(bounded) is not None
