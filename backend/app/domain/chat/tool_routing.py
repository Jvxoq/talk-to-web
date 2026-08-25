"""Whether a question is about the user's own uploaded files.

A pattern-matching rule rather than a judgement call, so it belongs here for
the same reason the guardrail detectors do: a policy expressed as plain stdlib
code, reviewable and unit-testable without a framework or a network call. The
application layer is what acts on the answer - see `ToolRegistry.invoke`, which
uses it to keep a web search from pre-empting a private lookup.

The patterns follow the same discipline as `guardrails.py`: linear in the
length of the input - no nested quantifiers, no alternation inside a repetition,
no unbounded lookaround - and bounded by `max_scan_chars`, because this runs on
the reply path of a single worker and the standard library offers no way to
time a regex match out.
"""

import re
from collections.abc import Sequence

# A question long enough to need more than this has said what it is about well
# before the cap. A user message is already capped at 32 KB by the request
# schema; this is the far tighter bound that keeps the scan's cost flat.
MAX_SCAN_CHARS = 4_000

# The nouns that name a thing a user hands over rather than publishes.
_DOC_NOUN = r"(?:documents?|docs?|files?|pdfs?|uploads?|attachments?)"

# The nouns that only count as "uploaded" when something says they were - a
# bare "the report" is as likely to be a public one, and a false positive
# costs the model a wasted round trip.
_QUALIFIED_NOUN = r"(?:report|contract|paper|memo|notes?|deck|slides?|spreadsheet|transcript)s?"

_HANDED_OVER = r"(?:uploaded|attached|shared|sent|provided|gave)"

# Each pattern is one shape, kept separate rather than joined into one
# alternation so that a wrong match is traceable to the rule that made it.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "my documents", "our uploaded files", "my own docs"
    re.compile(rf"\b(?:my|our)\s+(?:own\s+|uploaded\s+|attached\s+)?{_DOC_NOUN}\b", re.IGNORECASE),
    # "my uploaded report", "our attached contract" - the qualified nouns, which
    # need the qualifier to count.
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


# A document name has to be at least this long before it is worth matching on.
# Short stems collide with ordinary words - a file called "a.pdf" or "no.txt"
# would make every sentence document-scoped.
_MIN_NAME_CHARS = 4


def _names_pattern(document_names: Sequence[str]) -> re.Pattern[str] | None:
    """One pattern matching any of these documents by name, or `None`.

    Built per call rather than cached, because the names are one user's uploads
    and change as they upload. `re.escape` is not optional: a filename is user
    input, and an unescaped one is both a wrong match and a way to smuggle a
    pathological pattern into a scan that runs on the reply path.

    Both the full name and its stem are offered, so "what does budget-q3.pdf
    say" and "summarise budget-q3" both match the same upload.
    """
    forms: dict[str, None] = {}
    for name in document_names:
        stem = name.rpartition(".")[0] or name
        for candidate in (name, stem):
            if len(candidate) >= _MIN_NAME_CHARS:
                forms.setdefault(candidate, None)
    if not forms:
        return None
    # Longest first, so "budget-q3.pdf" wins over "budget-q3" and the match
    # reported is the most specific one. Alternation of literals only - no
    # nesting, no quantifier over a group - so this stays linear like the rest.
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])", re.IGNORECASE)


def is_document_scoped(
    text: str,
    *,
    document_names: Sequence[str] = (),
    max_scan_chars: int = MAX_SCAN_CHARS,
) -> bool:
    """Whether the user is asking about files they supplied.

    Deliberately conservative, and the asymmetry is the point: a false negative
    costs one web search that could have been a retrieval, while a false
    positive costs the model one redirect round trip on a question the search
    would have answered. Neither is fatal, but the first is the cheaper mistake
    to make often, so a phrase earns a match only when it names something
    handed over rather than published.

    `document_names` adds the one signal the patterns above cannot express:
    the user naming a file they actually own. "What does budget-q3.pdf say" and
    "summarise budget-q3" carry no possessive and no document noun, so nothing
    above fires, yet both are unmistakably about an upload. The list comes from
    the database rather than from the message, so it is not something the model
    can talk its way around.

    Note the deliberate limit: this matches the *name*, not the subject. Someone
    who uploads `budget-q3.pdf` and asks "what did we spend in Q3?" still
    matches nothing here, and that question is answered instead by the document
    digest `GenerateReply` puts in front of the model - knowing what the file is
    about is what makes it pick retrieval on its own. The two work as a pair:
    the digest informs the choice, this makes it binding.
    """
    if max_scan_chars <= 0:
        return False
    bounded = text[:max_scan_chars]
    if any(pattern.search(bounded) for pattern in _PATTERNS):
        return True
    names = _names_pattern(document_names)
    return names is not None and names.search(bounded) is not None
