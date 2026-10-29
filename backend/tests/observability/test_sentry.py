"""Error reporting: off without a DSN, and what it sends when on.

`sentry_sdk.init` is patched, so tests never install a real client.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import sentry_sdk
from pytest import MonkeyPatch

from app.observability.context import reset_request_id, set_request_id
from app.observability.sentry import _tag_request, configure_sentry, report_exception


@contextmanager
def serving(request_id: str) -> Iterator[None]:
    """Set a request id, and reset it after. Sync tests share one context."""
    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)


class TestConfigureSentry:
    @pytest.mark.parametrize("dsn", [None, ""])
    def test_no_dsn_means_no_initialization(self, dsn: str | None) -> None:
        """Both `None` and an empty `SENTRY_DSN=` leave reporting off."""
        assert (
            configure_sentry(dsn=dsn, environment="local", release=None, traces_sample_rate=0.0)
            is False
        )

    def test_a_dsn_initializes_the_sdk_without_pii(self, monkeypatch: MonkeyPatch) -> None:
        """Init never sends PII: no cookies, tokens or chat text."""
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
        """No request id means no tag."""
        assert _tag_request({}, {}) == {}

    def test_existing_tags_survive(self) -> None:
        with serving("abc123"):
            event = _tag_request({"tags": {"transaction": "POST /generate/text/"}}, {})

        assert event["tags"] == {"transaction": "POST /generate/text/", "request_id": "abc123"}


class TestReporting:
    def test_reporting_with_sentry_off_is_a_no_op_rather_than_an_error(self) -> None:
        """Without a DSN, reporting is a safe no-op."""
        report_exception(RuntimeError("boom"))
