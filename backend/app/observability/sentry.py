"""Error reporting. Initialized once, from the composition root.

Sentry is opt-in: with no DSN configured, `configure_sentry` does nothing and
`report_exception` is a no-op, so a local run and the test suite send nothing
anywhere and need no key. That is also why the rest of the codebase reports
through `report_exception` rather than importing `sentry_sdk` directly - one
import to remove if this is ever swapped for something else, and no vendor name
in the delivery layer.
"""

import sentry_sdk
from sentry_sdk.integrations.loguru import LoguruIntegration
from sentry_sdk.types import Event, Hint

from app.observability.context import get_request_id


def _tag_request(event: Event, _hint: Hint) -> Event:
    """Attach the correlation id, so an event and its log lines meet again.

    A `before_send` hook rather than a tag set per capture: it runs in the
    context of whatever raised, which covers the errors our own handler never
    sees - a failure inside a streaming response, or on a WebSocket.
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
    """Start reporting, if a DSN says where to. Returns whether it was enabled.

    Call before the `FastAPI` app is constructed: the SDK's integrations work by
    patching Starlette's classes at init time.
    """
    if not dsn:
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        # Unset locally; in a deployment it is the image tag or commit SHA, which
        # is what makes "this started at 14:02" answerable as "this started with
        # that deploy".
        release=release,
        traces_sample_rate=traces_sample_rate,
        # Off, deliberately. On, the SDK attaches cookies, the client IP and the
        # request body to every event - which for this app means the refresh
        # cookie, the bearer token and whatever the user typed into a chat.
        # Errors here are diagnosable without any of that.
        send_default_pii=False,
        integrations=[
            # Breadcrumbs from log lines, but no events from them. Without this,
            # `event_level=ERROR` turns every `logger.error` into its own issue -
            # including the one `app/api/errors.py` writes immediately before it
            # captures the exception properly, so each 500 would arrive twice,
            # once with a stack trace and once without.
            LoguruIntegration(event_level=None),
        ],
        before_send=_tag_request,
    )
    return True


def report_exception(exc: BaseException, *, request_id: str | None = None) -> None:
    """Send an exception to Sentry, if Sentry is on. A no-op if it is not.

    `request_id` is for the callers that know it and cannot rely on the hook
    above finding it - `app/api/errors.py` runs after the middleware has already
    unset the context variable. A fresh scope so the tag belongs to this event
    and not to everything else the request goes on to report.
    """
    if request_id is None:
        sentry_sdk.capture_exception(exc)
        return

    with sentry_sdk.new_scope() as scope:
        scope.set_tag("request_id", request_id)
        sentry_sdk.capture_exception(exc)
