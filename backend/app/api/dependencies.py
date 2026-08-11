"""Dependency providers for the API layer.

Every provider is a single lookup on the container assembled at the composition
root. A branch, an `await`, or any I/O in here means a use case is missing.

`get_current_user` is the one exception, and it is a deliberate one. It reads a
header and awaits `IdentifyRequest` - but that is a use case being *called*, the
same shape a route body has, not logic smuggled into a provider. Turning a
credential into a principal has to happen before routing reaches the handler,
which is precisely what a dependency is for. The authorization question - may
this principal touch this thing? - stays where it belongs, in the use cases, as
an owner threaded down to the query.
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
from app.application.ingestion.use_cases.index_document import IndexDocument
from app.application.ingestion.use_cases.ingest_url import IngestUrl
from app.application.ingestion.use_cases.list_documents import ListDocuments
from app.application.ingestion.use_cases.upload_document import UploadDocument
from app.application.transcription.use_cases.transcribe_stream import TranscribeStream
from app.domain.identity.errors import InvalidToken

_BEARER_PREFIX = "bearer "


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
    def list_conversations(self) -> ListConversations: ...

    @property
    def record_exchange(self) -> RecordExchange: ...

    @property
    def delete_conversation(self) -> DeleteConversation: ...

    @property
    def upload_document(self) -> UploadDocument: ...

    @property
    def ingest_url(self) -> IngestUrl: ...

    @property
    def index_document(self) -> IndexDocument: ...

    @property
    def list_documents(self) -> ListDocuments: ...

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
        """How to write the refresh cookie.

        Here for the same reason as `chat_models`: the API has to set the cookie
        and may not read settings to find out whether this deployment is
        cross-site, on https, or under a shared parent domain.
        """
        ...

    @property
    def trust_forwarded_client_ip(self) -> bool:
        """Whether `request.client` can be believed for rate limiting.

        A delivery fact, not a business one: it describes what sits in front of
        this process. Believing a proxy's own address would turn a per-caller
        limit into one shared bucket for everybody.
        """
        ...

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

    @property
    def websocket_origins(self) -> Sequence[str]:
        """Origins permitted to open the speech-to-text WebSocket.

        Here for the same reason as `chat_models`: refusing a handshake is a
        delivery concern, decided before any use case is reached, and the API
        layer may not import `app.settings` to find out who is allowed.
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


def get_list_conversations(container: ContainerDep) -> ListConversations:
    return container.list_conversations


def get_record_exchange(container: ContainerDep) -> RecordExchange:
    return container.record_exchange


def get_delete_conversation(container: ContainerDep) -> DeleteConversation:
    return container.delete_conversation


def get_upload_document(container: ContainerDep) -> UploadDocument:
    return container.upload_document


def get_ingest_url(container: ContainerDep) -> IngestUrl:
    return container.ingest_url


def get_index_document(container: ContainerDep) -> IndexDocument:
    return container.index_document


def get_list_documents(container: ContainerDep) -> ListDocuments:
    return container.list_documents


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
IngestUrlDep = Annotated[IngestUrl, Depends(get_ingest_url)]
IndexDocumentDep = Annotated[IndexDocument, Depends(get_index_document)]
ListDocumentsDep = Annotated[ListDocuments, Depends(get_list_documents)]
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
    """Pull the credential out of the Authorization header.

    Hand-read rather than taken from `fastapi.security.HTTPBearer`, which raises
    `HTTPException` directly and so would route a 401 around
    `app.api.errors` - the module that owns every status code in this app, and
    the reason a 401 carries `WWW-Authenticate` at all.
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
    """The caller's address, when the deployment is willing to vouch for it.

    Returned as `None` rather than as the proxy's own address when it is not:
    a rate limiter keyed on one shared value locks out every user at once, which
    is worse than not limiting by address at all. Uvicorn is what rewrites
    `connection.client` from `X-Forwarded-For`, and only when it was started
    with `--forwarded-allow-ips`; this flag is the operator confirming that it
    was.
    """
    if not trusted or connection.client is None:
        return None
    return connection.client.host


ClientIpDep = Annotated[str | None, Depends(client_ip)]
