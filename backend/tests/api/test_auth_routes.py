"""Auth routes and the gate in front of everything else.

Status codes, cookie attributes and the refusal of anonymous callers — not
business rules, which live in `tests/application/test_identity_use_cases.py`.
"""

from typing import Any, ClassVar

import pytest

from app.application.identity.dto import SessionTokens
from app.domain.identity.errors import InvalidCredentials, InvalidToken
from app.domain.usage.errors import RateLimited
from tests.api.test_routes import (
    AUTH,
    TOKEN,
    USER,
    StubContainer,
    StubUseCase,
    client,
    conversation,
    ws_client,
)

REFRESH_SECRET = "a-refresh-secret"


def tokens(refresh: str = REFRESH_SECRET) -> SessionTokens:
    return SessionTokens(
        access_token=TOKEN,
        expires_in=900,
        refresh_token=refresh,
        refresh_expires_in=1_209_600,
        user=USER,
    )


class TestRegisterAndLogin:
    async def test_register_returns_201_with_a_token_and_a_cookie(self) -> None:
        stub = StubUseCase(returns=tokens())

        async with client(StubContainer(register_user=stub)) as http:
            response = await http.post(
                "/auth/register", json={"email": "owner@example.com", "password": "x" * 12}
            )

        assert response.status_code == 201
        body = response.json()
        assert body["access_token"] == TOKEN
        assert body["token_type"] == "bearer"
        assert body["user"] == {"id": USER.user_id, "email": USER.email}
        assert response.cookies.get("refresh_token") == REFRESH_SECRET

    async def test_the_refresh_token_is_never_in_the_body(self) -> None:
        """The whole reason the pair is split across two channels.

        A script that manages to run on this origin can read the JSON. If the
        refresh token were in it, an XSS would buy a fortnight of session rather
        than the access token's few minutes.
        """
        async with client(StubContainer(authenticate_user=StubUseCase(returns=tokens()))) as http:
            response = await http.post(
                "/auth/login", json={"email": "owner@example.com", "password": "x" * 12}
            )

        assert REFRESH_SECRET not in response.text

    async def test_the_cookie_is_httponly_and_scoped_to_the_auth_routes(self) -> None:
        async with client(StubContainer(authenticate_user=StubUseCase(returns=tokens()))) as http:
            response = await http.post(
                "/auth/login", json={"email": "owner@example.com", "password": "x" * 12}
            )

        header = response.headers["set-cookie"].lower()
        # HttpOnly is what puts the token out of JavaScript's reach; the path is
        # what keeps it off every chat request and upload.
        assert "httponly" in header
        assert "path=/auth" in header

    async def test_bad_credentials_are_a_401_with_a_challenge(self) -> None:
        stub = StubUseCase(raises=InvalidCredentials())

        async with client(StubContainer(authenticate_user=stub)) as http:
            response = await http.post(
                "/auth/login", json={"email": "owner@example.com", "password": "x" * 12}
            )

        assert response.status_code == 401
        # RFC 9110 requires it, and a client cannot tell "sign in" from
        # "you may not" without it.
        assert response.headers["www-authenticate"] == "Bearer"

    async def test_too_many_attempts_is_a_429_that_says_for_how_long(self) -> None:
        stub = StubUseCase(raises=RateLimited(retry_after_seconds=42))

        async with client(StubContainer(authenticate_user=stub)) as http:
            response = await http.post(
                "/auth/login", json={"email": "owner@example.com", "password": "x" * 12}
            )

        assert response.status_code == 429
        assert response.headers["retry-after"] == "42"

    @pytest.mark.parametrize(
        "body",
        [
            {"email": "owner@example.com"},
            {"password": "x" * 12},
            {"email": "owner@example.com", "password": "x" * 12, "role": "admin"},
            {"email": "owner@example.com", "password": ""},
        ],
        ids=["no-password", "no-email", "extra-field", "empty-password"],
    )
    async def test_malformed_bodies_never_reach_the_use_case(self, body: dict[str, str]) -> None:
        # `extra="forbid"` is what makes the third case a 422: a request that
        # tries to set a field the schema does not have is rejected outright
        # rather than quietly ignored.
        stub = StubUseCase(returns=tokens())

        async with client(StubContainer(authenticate_user=stub)) as http:
            response = await http.post("/auth/login", json=body)

        assert response.status_code == 422
        assert stub.calls == []


class TestRefreshAndLogout:
    async def test_refresh_reads_the_cookie_and_writes_a_new_one(self) -> None:
        stub = StubUseCase(returns=tokens(refresh="rotated-secret"))

        async with client(StubContainer(refresh_session=stub)) as http:
            response = await http.post("/auth/refresh", cookies={"refresh_token": "old-secret"})

        assert response.status_code == 200
        assert stub.calls[0][0].refresh_token == "old-secret"
        assert response.cookies.get("refresh_token") == "rotated-secret"

    async def test_refresh_without_a_cookie_is_a_401_not_a_500(self) -> None:
        # This is the ordinary state of a first visit, and the frontend calls it
        # on every page load to find out whether it has a session.
        stub = StubUseCase(returns=tokens())

        async with client(StubContainer(refresh_session=stub)) as http:
            response = await http.post("/auth/refresh")

        assert response.status_code == 401
        assert stub.calls == []

    async def test_a_rejected_refresh_token_is_a_401(self) -> None:
        stub = StubUseCase(raises=InvalidToken("session was already used"))

        async with client(StubContainer(refresh_session=stub)) as http:
            response = await http.post("/auth/refresh", cookies={"refresh_token": "stale"})

        assert response.status_code == 401

    async def test_logout_revokes_and_clears_the_cookie(self) -> None:
        stub = StubUseCase()

        async with client(StubContainer(revoke_session=stub)) as http:
            response = await http.post("/auth/logout", cookies={"refresh_token": REFRESH_SECRET})

        assert response.status_code == 204
        assert stub.calls == [(REFRESH_SECRET,)]
        # Cleared with the same path it was written with - a mismatch leaves the
        # cookie in place and "signs out" into a session that still works.
        assert "path=/auth" in response.headers["set-cookie"].lower()

    async def test_logout_without_a_cookie_still_succeeds(self) -> None:
        stub = StubUseCase()

        async with client(StubContainer(revoke_session=stub)) as http:
            response = await http.post("/auth/logout")

        assert response.status_code == 204
        assert stub.calls == []


class TestMe:
    async def test_reports_the_signed_in_user(self) -> None:
        async with client(StubContainer()) as http:
            response = await http.get("/auth/me")

        assert response.status_code == 200
        assert response.json() == {"id": USER.user_id, "email": USER.email}

    async def test_is_a_401_without_a_token(self) -> None:
        async with client(StubContainer(), authenticated=False) as http:
            response = await http.get("/auth/me")

        assert response.status_code == 401


class TestEveryOtherRouteIsGated:
    """No route that touches a user's data may be reached anonymously.

    Parametrised over every one of them rather than spot-checked: the failure
    this guards against is a new route being added without the dependency, and
    only an exhaustive list notices that.
    """

    REQUESTS: ClassVar[list[tuple[str, str, dict[str, Any]]]] = [
        ("POST", "/generate/text/", {"json": {"model": "m", "user_input": "hi"}}),
        ("POST", "/conversations/", {"json": {"title": "T", "model_type": "m"}}),
        ("GET", "/conversations/1", {}),
        (
            "POST",
            "/conversations/1/messages",
            {"json": {"prompt_content": "p", "response_content": "r"}},
        ),
        ("POST", "/conversations/1/delete", {}),
        ("POST", "/upload/file/", {"files": {"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}}),
    ]

    @pytest.mark.parametrize(("method", "path", "kwargs"), REQUESTS)
    async def test_anonymous_requests_are_refused(
        self, method: str, path: str, kwargs: dict[str, Any]
    ) -> None:
        container = StubContainer(
            generate_reply=StubUseCase(),
            start_conversation=StubUseCase(returns=conversation()),
            get_conversation=StubUseCase(returns=conversation()),
            record_exchange=StubUseCase(),
            delete_conversation=StubUseCase(),
            upload_document=StubUseCase(),
            index_document=StubUseCase(),
        )

        async with client(container, authenticated=False) as http:
            response = await http.request(method, path, **kwargs)

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize(
        "header",
        [
            {"Authorization": "Bearer not-the-right-token"},
            {"Authorization": TOKEN},
            {"Authorization": f"Basic {TOKEN}"},
            {"Authorization": "Bearer"},
        ],
        ids=["wrong-token", "no-scheme", "wrong-scheme", "no-token"],
    )
    async def test_a_credential_that_is_not_a_valid_bearer_token_is_refused(
        self, header: dict[str, str]
    ) -> None:
        stub = StubUseCase(returns=conversation())

        async with client(StubContainer(get_conversation=stub), authenticated=False) as http:
            response = await http.get("/conversations/1", headers=header)

        assert response.status_code == 401
        assert stub.calls == []

    async def test_the_verified_user_is_what_reaches_the_use_case(self) -> None:
        """The client cannot choose whose conversations it reads.

        There is no owner field on the wire at all - it comes from the token - so
        this asserts the only thing that could go wrong: that the id threaded
        down is the token's.
        """
        stub = StubUseCase(returns=conversation())

        async with client(StubContainer(get_conversation=stub)) as http:
            await http.get("/conversations/1", headers=AUTH)

        assert stub.calls == [(1, USER.user_id)]


class TestTranscriptionSocketRequiresAToken:
    """The socket is authenticated before it is accepted.

    Every accepted socket opens a billed Deepgram session, so a caller who
    cannot prove who they are must be turned away at the handshake rather than
    after it.
    """

    class StubTranscribe:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, transport: Any) -> None:
            self.calls += 1
            await transport.ready()

    @pytest.mark.parametrize(
        "url",
        [
            "/ws/transcribe/",
            "/ws/transcribe/?token=",
            "/ws/transcribe/?token=not-a-real-token",
        ],
        ids=["no-token", "empty-token", "bad-token"],
    )
    def test_a_handshake_without_a_usable_token_never_opens_a_session(self, url: str) -> None:
        from starlette.websockets import WebSocketDisconnect

        stub = self.StubTranscribe()
        allowed = {"origin": "https://app.example.com"}

        with (
            ws_client(StubContainer(transcribe_stream=stub)) as socket_client,
            pytest.raises(WebSocketDisconnect) as refusal,
            socket_client.websocket_connect(url, headers=allowed),
        ):
            pass  # pragma: no cover - the handshake never gets this far

        assert refusal.value.code == 1008
        assert stub.calls == 0, "a refused handshake must never start a session"
