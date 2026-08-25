"""Which questions count as being about the user's own uploaded files.

Pure stdlib in, bool out: no fixtures, no async, no I/O.
"""

import pytest

from app.domain.chat.tool_routing import MAX_SCAN_CHARS, is_document_scoped


class TestDocumentScopedQuestions:
    @pytest.mark.parametrize(
        "text",
        [
            "Based on the documents I've uploaded, what year was Aurora founded?",
            "What does my PDF say about the warranty?",
            "According to my uploaded files, how many charge cycles is it rated for?",
            "Summarise the attachment for me.",
            "Can you check the doc I sent you?",
            "I uploaded a spec last week - what does it say about retries?",
            "What is in our attached contract?",
            "Pull the numbers out of the uploaded report.",
            "In the file I shared, who signed it?",
        ],
    )
    def test_a_question_naming_something_handed_over_is_document_scoped(self, text: str) -> None:
        assert is_document_scoped(text)


class TestEverythingElse:
    @pytest.mark.parametrize(
        "text",
        [
            "What is the weather in my city right now?",
            "What is today's top headline on the BBC News website?",
            "What is 12 times 12?",
            "Read https://example.com and tell me what it says.",
            "Who is the CEO of Aurora Robotics?",
            # "report" without a qualifier is as likely to be a public one, and
            # a false positive here would redirect a search that was right.
            "What did the IPCC report say about sea levels?",
            "Which file format does Kubernetes use for manifests?",
        ],
    )
    def test_a_question_about_the_world_is_not_document_scoped(self, text: str) -> None:
        assert not is_document_scoped(text)


class TestTheScanIsBounded:
    def test_a_phrase_past_the_cap_is_not_read(self) -> None:
        text = "x" * MAX_SCAN_CHARS + " what does my PDF say?"

        assert not is_document_scoped(text)
        assert is_document_scoped(text, max_scan_chars=len(text))

    def test_a_cap_of_zero_reads_nothing(self) -> None:
        assert not is_document_scoped("what does my PDF say?", max_scan_chars=0)


class TestAQuestionNamingAnOwnedDocument:
    """The gap the phrase patterns cannot close.

    Someone who uploads `budget-q3.pdf` and asks about it by name has used no
    possessive and no document noun, so nothing above fires. The list of files
    the account actually owns is what settles it - and it comes from the
    database, so it is not something the model can argue with.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "what does budget-q3.pdf say about travel?",
            "summarise budget-q3",
            "According to BUDGET-Q3.PDF, what changed?",
            "pull the headline number out of budget-q3.pdf",
        ],
    )
    def test_naming_the_file_is_document_scoped(self, text: str) -> None:
        assert is_document_scoped(text, document_names=["budget-q3.pdf"])

    def test_the_same_question_without_that_document_is_not(self) -> None:
        # The identical sentence, from an account that never uploaded it.
        assert not is_document_scoped("summarise budget-q3", document_names=["payroll.pdf"])

    def test_any_of_several_owned_documents_counts(self) -> None:
        names = ["payroll.pdf", "budget-q3.pdf", "roadmap.docx"]

        assert is_document_scoped("what is in roadmap.docx?", document_names=names)
        assert is_document_scoped("compare payroll and budget-q3", document_names=names)

    def test_no_documents_falls_back_to_the_phrase_patterns(self) -> None:
        assert not is_document_scoped("what did we spend?", document_names=[])
        assert is_document_scoped("what does my PDF say?", document_names=[])


class TestNameMatchingIsConservative:
    def test_a_name_must_be_a_whole_word(self) -> None:
        # "budget" appearing inside a longer token is not a reference to the
        # file - matching it would make half of every finance question private.
        assert not is_document_scoped("rebudgeting the quarter", document_names=["budget.pdf"])
        assert not is_document_scoped("budgetary policy", document_names=["budget.pdf"])

    @pytest.mark.parametrize("name", ["a.md", "no.txt", "q3.pdf", ".pdf"])
    def test_a_very_short_name_is_never_matched(self, name: str) -> None:
        # A two- or three-character stem collides with ordinary words, and a
        # file called `no.txt` would make every sentence containing "no"
        # document-scoped.
        assert not is_document_scoped("no, tell me about a q3 rebound", document_names=[name])

    def test_a_regex_special_filename_is_matched_literally(self) -> None:
        # An unescaped name here would be both a wrong match and a way to get a
        # pathological pattern onto the reply path, since the filename is user
        # input. It must be treated as text, and only as text.
        names = ["(a+)+b.pdf"]

        assert is_document_scoped("open (a+)+b.pdf", document_names=names)
        assert not is_document_scoped("open aaaaaaaaaaaaaaaaaaaaaaaab", document_names=names)

    def test_a_name_past_the_scan_cap_is_not_read(self) -> None:
        text = "x" * MAX_SCAN_CHARS + " what is in budget-q3.pdf?"

        assert not is_document_scoped(text, document_names=["budget-q3.pdf"])
        assert is_document_scoped(text, document_names=["budget-q3.pdf"], max_scan_chars=len(text))
