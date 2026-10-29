"""Dependency providers for the API layer.

Each provider is one container lookup. A branch, `await` or I/O here means a
use case is missing. `get_current_user` is the exception: it calls the
`IdentifyRequest` use case, the same way a route would.
"""

from collections.abc import Sequence
from typing import Annotated, Protocol

from fastapi import Depends
from starlette.requests import HTTPConnection

from app.application.chat.use_cases.delete_conversation import DeleteConversation
from app.application.chat.use_cases.generate_reply import GenerateReply
from app.application.chat.use_cases.get_conversation import GetConversation
from app.application.chat.use_cases.list_conversations import ListConversations
from app.application.chat.use_cases.record_exchange import RecordExchange
from app.application.chat.use_cases.start_conversation import StartConversation
from app.application.health.use_cases.check_readiness import CheckReadiness
from app.application.identity.dto import RefreshCookiePolicy, UserContext
from app.application.identity.use_cases.authenticate_user import AuthenticateUser
from app.application.identity.use_cases.identify_request import IdentifyRequest
from app.application.identity.use_cases.refresh_session import RefreshSession
from app.application.identity.use_cases.register_user import RegisterUser
from app.application.identity.use_cases.revoke_session import RevokeSession
from app.application.ingestion.use_cases.delete_document import DeleteDocument
from app.application.ingestion.use_cases.get_conversation_document import GetConversationDocument
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.list_documents import ListDocuments
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.application.transcription.use_cases.transcribe_stream import TranscribeStream
from app.domain.identity.errors import InvalidToken

_BEARER_PREFIX = "bearer "


class Container(Protocol):
    """What the API needs from the container.

    Declared here so the API never imports the composition root. Members are
    read-only properties, so a frozen dataclass can satisfy them.
    """

    @property
    def generate_reply(self) -> GenerateReply: ...

    @property
    def start_conversation(self) -> StartConversation: ...

    @property
    def get_conversation(self) -> GetConversation: ...

    @property
    def list_conversations(self) -> ListConversations: ...

    @property
    def record_exchange(self) -> RecordExchange: ...

    @property
    def delete_conversation(self) -> DeleteConversation: ...

    @property
    def upload_document(self) -> UploadDocument: ...

    @property
    def index_document(self) -> IndexDocument: ...

    @property
    def list_documents(self) -> ListDocuments: ...

    @property
    def get_conversation_document(self) -> GetConversationDocument: ...

    @property
    def delete_document(self) -> DeleteDocument: ...

    @property
    def transcribe_stream(self) -> TranscribeStream: ...

    @property
    def register_user(self) -> RegisterUser: ...

    @property
    def authenticate_user(self) -> AuthenticateUser: ...

    @property
    def refresh_session(self) -> RefreshSession: ...

    @property
    def revoke_session(self) -> RevokeSession: ...

    @property
    def identify_request(self) -> IdentifyRequest: ...

    @property
    def check_readiness(self) -> CheckReadiness: ...

    @property
    def refresh_cookie(self) -> RefreshCookiePolicy:
        """How to write the refresh cookie. Config the API may not read from settings."""
        ...

    @property
    def trust_forwarded_client_ip(self) -> bool:
        """Whether `request.client` is the real caller, for rate limiting."""
        ...

    @property
    def chat_models(self) -> Sequence[str]:
        """Models this deployment accepts, default first.

        Used to reject unknown models and to list them for the frontend.
        """
        ...

    @property
    def websocket_origins(self) -> Sequence[str]:
        """Origins allowed to open the speech-to-text WebSocket."""
        ...


def get_container(connection: HTTPConnection) -> Container:
    """Read the container off the running app.

    Takes `HTTPConnection`, because a `Request` parameter fails on WebSocket routes.
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


def get_list_conversations(container: ContainerDep) -> ListConversations:
    return container.list_conversations


def get_record_exchange(container: ContainerDep) -> RecordExchange:
    return container.record_exchange


def get_delete_conversation(container: ContainerDep) -> DeleteConversation:
    return container.delete_conversation


def get_upload_document(container: ContainerDep) -> UploadDocument:
    return container.upload_document


def get_index_document(container: ContainerDep) -> IndexDocument:
    return container.index_document


def get_list_documents(container: ContainerDep) -> ListDocuments:
    return container.list_documents


def get_get_conversation_document(container: ContainerDep) -> GetConversationDocument:
    return container.get_conversation_document


def get_delete_document(container: ContainerDep) -> DeleteDocument:
    return container.delete_document


def get_transcribe_stream(container: ContainerDep) -> TranscribeStream:
    return container.transcribe_stream


def get_register_user(container: ContainerDep) -> RegisterUser:
    return container.register_user


def get_authenticate_user(container: ContainerDep) -> AuthenticateUser:
    return container.authenticate_user


def get_refresh_session(container: ContainerDep) -> RefreshSession:
    return container.refresh_session


def get_revoke_session(container: ContainerDep) -> RevokeSession:
    return container.revoke_session


def get_identify_request(container: ContainerDep) -> IdentifyRequest:
    return container.identify_request


def get_check_readiness(container: ContainerDep) -> CheckReadiness:
    return container.check_readiness


def get_refresh_cookie(container: ContainerDep) -> RefreshCookiePolicy:
    return container.refresh_cookie


def get_trust_forwarded_client_ip(container: ContainerDep) -> bool:
    return container.trust_forwarded_client_ip


def get_chat_models(container: ContainerDep) -> Sequence[str]:
    return container.chat_models


def get_websocket_origins(container: ContainerDep) -> Sequence[str]:
    return container.websocket_origins


GenerateReplyDep = Annotated[GenerateReply, Depends(get_generate_reply)]
StartConversationDep = Annotated[StartConversation, Depends(get_start_conversation)]
GetConversationDep = Annotated[GetConversation, Depends(get_get_conversation)]
ListConversationsDep = Annotated[ListConversations, Depends(get_list_conversations)]
RecordExchangeDep = Annotated[RecordExchange, Depends(get_record_exchange)]
DeleteConversationDep = Annotated[DeleteConversation, Depends(get_delete_conversation)]
UploadDocumentDep = Annotated[UploadDocument, Depends(get_upload_document)]
IndexDocumentDep = Annotated[IndexDocument, Depends(get_index_document)]
ListDocumentsDep = Annotated[ListDocuments, Depends(get_list_documents)]
GetConversationDocumentDep = Annotated[
    GetConversationDocument, Depends(get_get_conversation_document)
]
DeleteDocumentDep = Annotated[DeleteDocument, Depends(get_delete_document)]
TranscribeStreamDep = Annotated[TranscribeStream, Depends(get_transcribe_stream)]
ChatModelsDep = Annotated[Sequence[str], Depends(get_chat_models)]
WebSocketOriginsDep = Annotated[Sequence[str], Depends(get_websocket_origins)]
RegisterUserDep = Annotated[RegisterUser, Depends(get_register_user)]
AuthenticateUserDep = Annotated[AuthenticateUser, Depends(get_authenticate_user)]
RefreshSessionDep = Annotated[RefreshSession, Depends(get_refresh_session)]
RevokeSessionDep = Annotated[RevokeSession, Depends(get_revoke_session)]
IdentifyRequestDep = Annotated[IdentifyRequest, Depends(get_identify_request)]
CheckReadinessDep = Annotated[CheckReadiness, Depends(get_check_readiness)]
RefreshCookieDep = Annotated[RefreshCookiePolicy, Depends(get_refresh_cookie)]
TrustForwardedClientIpDep = Annotated[bool, Depends(get_trust_forwarded_client_ip)]


def bearer_token(connection: HTTPConnection) -> str:
    """Read the bearer token from the Authorization header.

    Not `HTTPBearer`, which raises `HTTPException` and skips `app.api.errors`.
    """
    header = connection.headers.get("authorization", "")
    if header[: len(_BEARER_PREFIX)].lower() != _BEARER_PREFIX:
        raise InvalidToken("expected an Authorization: Bearer header")
    return header[len(_BEARER_PREFIX) :].strip()


async def get_current_user(
    token: Annotated[str, Depends(bearer_token)],
    identify: IdentifyRequestDep,
) -> UserContext:
    """Who is making this request. Raises `InvalidToken` / `TokenExpired` if nobody."""
    return await identify(token)


CurrentUserDep = Annotated[UserContext, Depends(get_current_user)]


def client_ip(connection: HTTPConnection, trusted: TrustForwardedClientIpDep) -> str | None:
    """The caller's IP when trusted, else `None`.

    `None`, not the proxy's IP, which would put every user in one rate-limit
    bucket. Uvicorn sets the real IP only with `--forwarded-allow-ips`.
    """

    if not trusted or connection.client is None:
        return None
    return connection.client.host


ClientIpDep = Annotated[str | None, Depends(client_ip)]
