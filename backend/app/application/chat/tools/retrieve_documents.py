"""The agent's window onto the user's own uploaded documents."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.application.chat.models import Source
from app.application.chat.ports import KnowledgeRetriever
from app.application.chat.tools.base import BaseTool, ToolContext, ToolResult


class RetrieveDocumentsArgs(BaseModel):
    """What the model must supply to search the knowledge base."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description=(
            "What to look for, written as the user's question or as the key terms "
            "from it. Retrieval is by meaning, not keyword, so a full question "
            "works better than a bag of words."
        ),
    )


class RetrieveDocuments(BaseTool[RetrieveDocumentsArgs]):
    """
    Searches the passages indexed from the files the user uploaded.

    Takes `KnowledgeRetriever` and nothing else: the tool has no idea an
    embedding model or a vector store is involved, which is what lets either be
    replaced without touching this file.

    "The user's documents" in the description below is load-bearing, not
    marketing: the owner comes from the run context, never from the model, so
    the only documents this can reach are the ones the person asking uploaded.
    The conversation comes from the same place and narrows it further, to the
    files attached to this thread - a file the same person uploaded in another
    chat is out of reach here, which is what stops one document bleeding into
    an unrelated conversation.
    """

    name: ClassVar[str] = "retrieve_documents"
    description: ClassVar[str] = (
        "Search the documents this user has uploaded and return the passages that "
        "match. Use this first for any question that refers to 'my document', "
        "'the PDF', 'the file I uploaded', a report, a contract, or anything else "
        "the user appears to have supplied rather than published on the web - and "
        "also for any question naming a specific person, company, product or topic "
        "you don't already know, even without that phrasing, since the answer may "
        "be in a file the user uploaded. This searches private uploaded files only "
        "- it cannot see the public internet."
    )
    args_model: ClassVar[type[BaseModel]] = RetrieveDocumentsArgs

    def __init__(self, knowledge: KnowledgeRetriever) -> None:
        self._knowledge = knowledge

    async def _run(self, args: RetrieveDocumentsArgs, context: ToolContext) -> ToolResult:
        passages = await self._knowledge.retrieve(
            args.query, context.owner_id, context.conversation_id
        )
        if not passages:
            # An empty string reads to the model as a broken tool, and it tends
            # to retry the same call. A sentence saying "nothing matched" is what
            # tells it to answer from what it already knows instead.
            return ToolResult(
                content=(
                    f"No passages in the uploaded documents matched {args.query!r}. "
                    "Nothing relevant has been uploaded, or the wording is too far "
                    "from the text. If this is about current or public information, "
                    "try search_web next; otherwise answer from what you already know."
                )
            )

        # Deduplicated, in first-seen order: several passages routinely come
        # from the same document, and the citation list is about which
        # documents grounded the answer, not how many chunks each contributed.
        seen: dict[str, None] = {}
        for passage in passages:
            seen.setdefault(passage.source, None)

        return ToolResult(
            content="\n\n".join(passage.text for passage in passages),
            sources=tuple(Source(label=name) for name in seen),
        )
