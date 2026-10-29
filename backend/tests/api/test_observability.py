"""Request ids, log lines and Sentry reports, on the real app with stubs.

Covers where the id appears and that unhandled errors are reported.
"""

import json
import re

from loguru import logger
from pytest import MonkeyPatch

from app.api import errors
from app.domain.chat.errors import ConversationNotFound
from app.observability.context import (
    NO_REQUEST_ID,
    REQUEST_ID_HEADER,
    get_request_id,
)
from tests.api.test_routes import StubContainer, StubUseCase, client

GENERATED = re.compile(r"^[0-9a-f]{32}$")


class TestRequestIdHeader:
    async def test_every_response_carries_a_generated_id(self) -> None:
        async with client(StubContainer()) as http:
            response = await http.get("/health")

        assert GENERATED.match(response.headers[REQUEST_ID_HEADER])

    async def test_two_requests_get_two_ids(self) -> None:
        async with client(StubContainer()) as http:
            first = await http.get("/health")
            second = await http.get("/health")

        assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]

    async def test_an_id_from_upstream_is_adopted_so_one_id_spans_both_hops(self) -> None:
        async with client(StubContainer()) as http:
            response = await http.get("/health", headers={REQUEST_ID_HEADER: "caddy-0001"})

        assert response.headers[REQUEST_ID_HEADER] == "caddy-0001"

    async def test_a_malformed_id_is_replaced_rather_than_echoed(self) -> None:
        """A malformed incoming id is replaced, so it cannot inject headers or logs."""
        async with client(StubContainer()) as http:
            response = await http.get("/health", headers={REQUEST_ID_HEADER: "a b" + "!" * 200})

        assert GENERATED.match(response.headers[REQUEST_ID_HEADER])

    async def test_an_error_response_carries_one_too(self) -> None:
        stub = StubUseCase(raises=ConversationNotFound(99))

        async with client(StubContainer(get_conversation=stub)) as http:
            response = await http.get("/conversations/99")

        assert response.status_code == 404
        assert GENERATED.match(response.headers[REQUEST_ID_HEADER])

    async def test_the_id_does_not_outlive_the_request(self) -> None:
        """The id is reset after the request."""
        async with client(StubContainer()) as http:
            await http.get("/health")

        assert get_request_id() is None


class TestRequestIdInTheBody:
    async def test_a_500_body_quotes_the_same_id_as_the_header(self) -> None:
        """A 500 body carries the request id."""
        stub = StubUseCase(raises=RuntimeError("psycopg: password authentication failed"))

        async with client(StubContainer(get_conversation=stub), reraise=False) as http:
            response = await http.get("/conversations/1")

        assert response.status_code == 500
        assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]

    async def test_a_4xx_body_does_not(self) -> None:
        """Nothing to correlate: the client is being told exactly what was wrong."""
        stub = StubUseCase(raises=ConversationNotFound(99))

        async with client(StubContainer(get_conversation=stub)) as http:
            response = await http.get("/conversations/99")

        assert "request_id" not in response.json()


class TestRequestIdInLogs:
    """Sinks are added after the client, since `create_app` resets logging."""

    async def test_a_line_logged_while_serving_carries_the_request_id(self) -> None:
        """Log lines carry the incoming request id."""
        stub = StubUseCase(raises=RuntimeError("psycopg: password authentication failed"))
        written: list[str] = []

        async with client(StubContainer(get_conversation=stub), reraise=False) as http:
            sink = logger.add(written.append, serialize=True)
            try:
                response = await http.get("/conversations/1")
            finally:
                logger.remove(sink)

        # The handler's error line.
        logged = [json.loads(line)["record"]["extra"]["request_id"] for line in written]
        assert logged == [response.headers[REQUEST_ID_HEADER]]

    async def test_a_line_logged_outside_a_request_says_so(self) -> None:
        """Outside a request, the field is still present."""
        written: list[str] = []

        async with client(StubContainer()):
            sink = logger.add(written.append, serialize=True)
            try:
                logger.info("no request in scope")
            finally:
                logger.remove(sink)

        assert json.loads(written[0])["record"]["extra"]["request_id"] == NO_REQUEST_ID


class TestErrorReporting:
    """Which failures get reported. `report_exception` is patched in `errors.py`."""

    async def test_an_unhandled_error_is_reported_not_just_logged(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        reported: list[tuple[BaseException, str | None]] = []
        monkeypatch.setattr(
            errors,
            "report_exception",
            lambda exc, *, request_id=None: reported.append((exc, request_id)),
        )
        boom = RuntimeError("boom")

        async with client(
            StubContainer(get_conversation=StubUseCase(raises=boom)), reraise=False
        ) as http:
            response = await http.get("/conversations/1")

        # Tagged with the request id.
        assert reported == [(boom, response.headers[REQUEST_ID_HEADER])]

    async def test_a_mapped_domain_error_is_not(self, monkeypatch: MonkeyPatch) -> None:
        """A 404 is not reported."""
        reported: list[tuple[BaseException, str | None]] = []
        monkeypatch.setattr(
            errors,
            "report_exception",
            lambda exc, *, request_id=None: reported.append((exc, request_id)),
        )

        async with client(
            StubContainer(get_conversation=StubUseCase(raises=ConversationNotFound(99)))
        ) as http:
            await http.get("/conversations/99")

        assert reported == []
