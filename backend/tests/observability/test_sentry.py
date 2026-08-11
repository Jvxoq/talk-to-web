"""Error reporting: that it stays off without a DSN, and what it sends when on.

Nothing here calls `sentry_sdk.init` for real — that installs a client on a
process-global hub for the rest of the session, and a test suite that can phone
home is a test suite that eventually will. `init` is patched out instead, and
the event hook is exercised directly.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import sentry_sdk
from pytest import MonkeyPatch

from app.observability.context import reset_request_id, set_request_id
from app.observability.sentry import _tag_request, configure_sentry, report_exception


@contextmanager
def serving(request_id: str) -> Iterator[None]:
    """A request in scope, undone afterwards.

    These are synchronous tests, so they share one context: a `ContextVar` left
    set here is still set in the next test in the file.
    """
    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)


class TestConfigureSentry:
    def test_no_dsn_means_no_initialization(self) -> None:
        """The local and CI default. Reporting must be something a deployment
        opts into, not something a missing variable half-enables."""
        assert (
            configure_sentry(dsn=None, environment="local", release=None, traces_sample_rate=0.0)
            is False
        )

    def test_an_empty_dsn_is_treated_as_no_dsn(self) -> None:
        """`SENTRY_DSN=` in an env file is how "off" is usually spelled by
        accident; the SDK would take the empty string as a configuration error."""
        assert (
            configure_sentry(dsn="", environment="local", release=None, traces_sample_rate=0.0)
            is False
        )

    def test_a_dsn_initializes_the_sdk_without_pii(self, monkeypatch: MonkeyPatch) -> None:
        """The one thing worth asserting about the init call: it does not ship
        cookies, bearer tokens or chat messages with every event."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: captured.update(kwargs))

        enabled = configure_sentry(
            dsn="https://key@example.ingest.sentry.io/1",
            environment="production",
            release="abc123",
            traces_sample_rate=0.1,
        )

        assert enabled is True
        assert captured["send_default_pii"] is False
        assert captured["environment"] == "production"
        assert captured["release"] == "abc123"
        assert captured["traces_sample_rate"] == 0.1


class TestTagging:
    def test_an_event_is_tagged_with_the_request_it_came_from(self) -> None:
        with serving("abc123"):
            event = _tag_request({}, {})

        assert event["tags"]["request_id"] == "abc123"

    def test_an_event_raised_outside_a_request_is_left_untagged(self) -> None:
        """Startup and shutdown errors are real errors, and an empty or invented
        tag on them would only make the field unsearchable."""
        assert _tag_request({}, {}) == {}

    def test_existing_tags_survive(self) -> None:
        with serving("abc123"):
            event = _tag_request({"tags": {"transaction": "POST /generate/text/"}}, {})

        assert event["tags"] == {"transaction": "POST /generate/text/", "request_id": "abc123"}


class TestReporting:
    def test_reporting_with_sentry_off_is_a_no_op_rather_than_an_error(self) -> None:
        """Every 500 goes through this. It must not turn a handled failure into
        a second, unhandled one on a deployment that never configured a DSN."""
        report_exception(RuntimeError("boom"))
