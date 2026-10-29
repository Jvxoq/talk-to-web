"""Error reporting. Initialized once, from the composition root.

Without a DSN, everything here is a no-op. Nothing else imports `sentry_sdk`.
"""

import sentry_sdk
from sentry_sdk.integrations.loguru import LoguruIntegration
from sentry_sdk.types import Event, Hint

from app.observability.context import get_request_id


def _tag_request(event: Event, _hint: Hint) -> Event:
    """Tag the event with the request id.

    A `before_send` hook, so it also covers errors in streams and WebSockets.
    """
    request_id = get_request_id()
    if request_id is not None:
        event.setdefault("tags", {})["request_id"] = request_id
    return event


def configure_sentry(
    *,
    dsn: str | None,
    environment: str,
    release: str | None,
    traces_sample_rate: float,
) -> bool:
    """Start reporting if a DSN is set. Returns whether it was enabled.

    Call before building the `FastAPI` app, since the SDK patches Starlette at init.
    """
    if not dsn:
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        # The image tag or commit SHA, so errors map to deploys.
        release=release,
        traces_sample_rate=traces_sample_rate,
        # Off: it would send cookies, tokens and chat text.
        send_default_pii=False,
        integrations=[
            # Log lines become breadcrumbs, not events, so each 500 is reported once.
            LoguruIntegration(event_level=None),
        ],
        before_send=_tag_request,
    )
    return True


def report_exception(exc: BaseException, *, request_id: str | None = None) -> None:
    """Send an exception to Sentry, if enabled.

    Pass `request_id` when the context variable is already reset, as in
    `app/api/errors.py`. A fresh scope keeps the tag on this event only.
    """

    if request_id is None:
        sentry_sdk.capture_exception(exc)
        return

    with sentry_sdk.new_scope() as scope:
        scope.set_tag("request_id", request_id)
        sentry_sdk.capture_exception(exc)
