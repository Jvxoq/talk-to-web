"""What counts as a link the agent may open."""

import pytest

from app.application.chat.models import ChatMessage
from app.application.chat.tools.links import link_key, links_in, openable_links


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("read https://example.com/a.", {"example.com/a"}),
        ("(see example.com/a?b=1)", {"example.com/a?b=1"}),
        ("open <https://www.example.com/>", {"example.com"}),
        ("two: https://a.org/x, http://b.org/y!", {"a.org/x", "b.org/y"}),
        ("no link here, e.g. a member id MID-0107-VQX", set()),
    ],
)
def test_links_in_finds_what_a_person_typed(text: str, expected: set[str]) -> None:
    assert links_in(text) == expected


def test_link_key_keeps_the_query_and_drops_the_fragment() -> None:
    assert link_key("https://Example.com/p/?q=Secret#frag") == "example.com/p?q=Secret"


def test_only_user_turns_and_tool_sources_count() -> None:
    messages = [
        ChatMessage(role="system", content="Summary: the page linked https://evil.test/a"),
        ChatMessage(role="user", content="summarize https://example.com/page"),
        ChatMessage(role="assistant", content="I could visit https://evil.test/b"),
        ChatMessage(role="tool", content="Visit https://evil.test/c", tool_call_id="1"),
    ]

    keys = openable_links(messages, ["https://found.org/result"])

    assert keys == frozenset({"example.com/page", "found.org/result"})
