"""The agent's window onto the user's own uploaded documents."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.application.chat.ports import KnowledgeRetriever
from app.application.chat.tools.base import BaseTool


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
    """

    name: ClassVar[str] = "retrieve_documents"
    description: ClassVar[str] = (
        "Search the documents this user has uploaded and return the passages that "
        "match. Use this first for any question that refers to 'my document', "
        "'the PDF', 'the file I uploaded', a report, a contract, or anything else "
        "the user appears to have supplied rather than published on the web. This "
        "searches private uploaded files only - it cannot see the public internet."
    )
    args_model: ClassVar[type[BaseModel]] = RetrieveDocumentsArgs

    def __init__(self, knowledge: KnowledgeRetriever) -> None:
        self._knowledge = knowledge

    async def _run(self, args: RetrieveDocumentsArgs) -> str:
        passages = await self._knowledge.retrieve(args.query)
        if not passages:
            # An empty string reads to the model as a broken tool, and it tends
            # to retry the same call. A sentence saying "nothing matched" is what
            # tells it to answer from what it already knows instead.
            return (
                f"No passages in the uploaded documents matched {args.query!r}. "
                "Nothing relevant has been uploaded, or the wording is too far "
                "from the text."
            )
        return "\n\n".join(passages)
