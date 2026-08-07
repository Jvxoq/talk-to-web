"""Value objects for the chat context: the rules for reading what a user said.

Composing the model's prompt used to live here too. It does not any more: the
agent builds a message list and lets the model decide what context it needs,
so there is no single "final text" left to assemble.
"""

import re
from dataclasses import dataclass

from app.domain.chat.errors import EmptyUserMessage

_URL_PATTERN = re.compile(r"https?://[^\s]+")

# Trailing punctuation is almost always sentence punctuation rather than part of
# the address, and a stray "." or "," turns a good fetch into a 404.
_URL_TRAILING_NOISE = ".,;:!?\"'"

# A closing bracket is the exception: it may genuinely belong to the address.
# Wikipedia disambiguates titles with them - "/wiki/Hexagonal_architecture_(software)"
# - and stripping that ")" unconditionally requested a page that does not exist,
# so every such link silently contributed nothing. A closer is only noise when
# nothing earlier in the URL opened it.
_URL_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}


def _trim_trailing_noise(url: str) -> str:
    """Drop the sentence punctuation a URL collected, keeping balanced brackets."""
    while url:
        last = url[-1]
        if last in _URL_TRAILING_NOISE:
            url = url[:-1]
            continue

        opener = _URL_BRACKET_PAIRS.get(last)
        if opener is not None and url.count(last) > url.count(opener):
            url = url[:-1]
            continue

        break
    return url


@dataclass(frozen=True, slots=True)
class UserMessage:
    """One thing the user typed, with the rules for reading it."""

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise EmptyUserMessage()

    def urls(self) -> tuple[str, ...]:
        """Every http(s) address mentioned, in the order they were written."""
        found = (_trim_trailing_noise(url) for url in _URL_PATTERN.findall(self.text))
        return tuple(url for url in found if url)
