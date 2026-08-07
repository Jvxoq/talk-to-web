"""Domain rules, tested with no fixtures, no async, and no I/O."""

import pytest

from app.domain.chat.entities import Conversation, Message
from app.domain.chat.errors import EmptyUserMessage
from app.domain.chat.value_objects import UserMessage
from app.domain.ingestion.entities import Document
from app.domain.ingestion.errors import UnsupportedDocumentType
from app.domain.ingestion.value_objects import DocumentName
from app.domain.transcription.entities import AudioFormat, Transcript


class TestUserMessage:
    def test_rejects_blank_text(self) -> None:
        with pytest.raises(EmptyUserMessage):
            UserMessage("   \n ")

    def test_finds_urls_in_order(self) -> None:
        message = UserMessage("read https://a.dev/x then http://b.dev/y please")
        assert message.urls() == ("https://a.dev/x", "http://b.dev/y")

    def test_strips_sentence_punctuation_from_urls(self) -> None:
        assert UserMessage("see https://a.dev/page.").urls() == ("https://a.dev/page",)
        assert UserMessage("(https://a.dev/p)").urls() == ("https://a.dev/p",)

    def test_no_urls_is_empty(self) -> None:
        assert UserMessage("just a question").urls() == ()


class TestDocumentName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "passwd"),
            ("a/b/c/report.PDF", "report.PDF"),
            ("my file (1).pdf", "my_file__1_.pdf"),
        ],
    )
    def test_sanitizes_to_a_bare_filename(self, raw: str, expected: str) -> None:
        assert DocumentName.sanitize(raw).value == expected

    @pytest.mark.parametrize("raw", [None, "", "...", "/", "///"])
    def test_rejects_names_that_are_only_path(self, raw: str | None) -> None:
        with pytest.raises(UnsupportedDocumentType):
            DocumentName.sanitize(raw)

    def test_bounds_an_absurdly_long_name(self) -> None:
        assert len(DocumentName.sanitize("x" * 500 + ".pdf").value) <= 115


class TestDocumentChunking:
    def test_short_document_is_one_chunk(self) -> None:
        chunks = list(Document(DocumentName("a.pdf"), "hello there").chunks(500, 50))
        assert [c.text for c in chunks] == ["hello there"]
        assert chunks[0].source == "a.pdf"

    def test_blank_document_yields_nothing(self) -> None:
        assert list(Document(DocumentName("a.pdf"), "   \n  ").chunks(500, 50)) == []

    def test_windows_overlap_so_boundaries_stay_retrievable(self) -> None:
        text = "".join(str(i % 10) for i in range(250))
        chunks = [c.text for c in Document(DocumentName("a.pdf"), text).chunks(100, 20)]

        # stride 80 -> windows at 0, 80, 160; the last one runs to the end
        # rather than leaving a stub behind.
        assert len(chunks) == 3
        assert chunks[0][-20:] == chunks[1][:20]
        assert chunks[-1] == text[160:]

    def test_covers_every_character(self) -> None:
        text = "abcdefghij" * 30
        chunks = [c.text for c in Document(DocumentName("a.pdf"), text).chunks(100, 25)]
        assert chunks[0].startswith("abc")
        assert chunks[-1].endswith("hij")

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1), (100, 150)])
    def test_rejects_impossible_windows(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            list(Document(DocumentName("a.pdf"), "text").chunks(size, overlap))

    def test_clean_collapses_extraction_debris(self) -> None:
        assert Document.clean("a\n\nb   c. ,d..e") == "a b cd.e"


class TestConversation:
    def test_record_attaches_the_message(self) -> None:
        conversation = Conversation(title="t", model_type="m", id=7)
        message = conversation.record(Message(prompt_content="p", response_content="r"))
        assert conversation.messages == [message]
        assert message.conversation_id == 7

    def test_token_total_is_derived_when_absent(self) -> None:
        message = Message("p", "r", prompt_tokens=10, response_tokens=5)
        assert message.token_total() == 15

    def test_token_total_is_none_when_nothing_reported(self) -> None:
        assert Message("p", "r").token_total() is None


class TestTranscription:
    def test_blank_transcript_is_not_worth_sending(self) -> None:
        assert not Transcript(text="  ", is_final=True).is_worth_sending()
        assert Transcript(text="hi", is_final=False).is_worth_sending()

    @pytest.mark.parametrize("rate", [10, 0, 1_000_000])
    def test_rejects_implausible_sample_rates(self, rate: int) -> None:
        with pytest.raises(ValueError):
            AudioFormat(sample_rate=rate)
