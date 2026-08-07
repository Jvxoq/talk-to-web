"""Dependency providers for the API layer.

Every provider is a single lookup on the container assembled at the composition
root. A branch, an `await`, or any I/O in here means a use case is missing.
"""

from collections.abc import Sequence
from typing import Annotated, Protocol

from fastapi import Depends
from starlette.requests import HTTPConnection

from app.application.chat.use_cases.delete_conversation import DeleteConversation
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.application.chat.use_cases.get_conversation import GetConversation
from app.application.chat.use_cases.record_exchange import RecordExchange
from app.application.chat.use_cases.start_conversation import StartConversation
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.application.transcription.use_cases.transcribe_stream import TranscribeStream


class Container(Protocol):
    """
    What the API needs the container to provide.

    Declared here rather than imported from `app.composition`, because naming
    the concrete `AppContainer` — even behind `TYPE_CHECKING` — points this
    package at the composition root, and through it at every adapter. The real
    container satisfies this structurally; mypy checks that it does, and a test
    can supply anything else with the same attributes.

    Members are properties, not plain attributes: a protocol attribute is
    settable and therefore invariant, which a frozen container could never
    satisfy. Read-only is also the honest shape — nothing here rebinds them.
    """

    @property
    def generate_reply(self) -> GenerateReply: ...

    @property
    def start_conversation(self) -> StartConversation: ...

    @property
    def get_conversation(self) -> GetConversation: ...

    @property
    def record_exchange(self) -> RecordExchange: ...

    @property
    def delete_conversation(self) -> DeleteConversation: ...

    @property
    def upload_document(self) -> UploadDocument: ...

    @property
    def index_document(self) -> IndexDocument: ...

    @property
    def transcribe_stream(self) -> TranscribeStream: ...

    @property
    def chat_models(self) -> Sequence[str]:
        """The models this deployment will answer on, best-default first.

        Not a use case, and the one piece of configuration the API is allowed to
        see. It has to be here because two delivery concerns need it - rejecting
        an unknown model with a 4xx, and telling the frontend what to offer - and
        the API layer may not import `app.settings`. Arriving through the
        container keeps settings entering at the composition root only.
        """
        ...


def get_container(connection: HTTPConnection) -> Container:
    """Read the container off the running app.

    Typed as `HTTPConnection`, the common base of `Request` and `WebSocket`,
    because this provider serves both. FastAPI only fills a `Request` parameter
    on an HTTP route; asking for one from the `/ws/transcribe/` handler left the
    argument unbound and failed the connection with a 500 before the socket was
    ever usable. `HTTPConnection` is populated in both scopes.
    """
    container: Container = connection.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_generate_reply(container: ContainerDep) -> GenerateReply:
    return container.generate_reply


def get_start_conversation(container: ContainerDep) -> StartConversation:
    return container.start_conversation


def get_get_conversation(container: ContainerDep) -> GetConversation:
    return container.get_conversation


def get_record_exchange(container: ContainerDep) -> RecordExchange:
    return container.record_exchange


def get_delete_conversation(container: ContainerDep) -> DeleteConversation:
    return container.delete_conversation


def get_upload_document(container: ContainerDep) -> UploadDocument:
    return container.upload_document


def get_index_document(container: ContainerDep) -> IndexDocument:
    return container.index_document


def get_transcribe_stream(container: ContainerDep) -> TranscribeStream:
    return container.transcribe_stream


def get_chat_models(container: ContainerDep) -> Sequence[str]:
    return container.chat_models


GenerateReplyDep = Annotated[GenerateReply, Depends(get_generate_reply)]
StartConversationDep = Annotated[StartConversation, Depends(get_start_conversation)]
GetConversationDep = Annotated[GetConversation, Depends(get_get_conversation)]
RecordExchangeDep = Annotated[RecordExchange, Depends(get_record_exchange)]
DeleteConversationDep = Annotated[DeleteConversation, Depends(get_delete_conversation)]
UploadDocumentDep = Annotated[UploadDocument, Depends(get_upload_document)]
IndexDocumentDep = Annotated[IndexDocument, Depends(get_index_document)]
TranscribeStreamDep = Annotated[TranscribeStream, Depends(get_transcribe_stream)]
ChatModelsDep = Annotated[Sequence[str], Depends(get_chat_models)]
