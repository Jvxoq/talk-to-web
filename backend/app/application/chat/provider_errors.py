"""Recognising a provider's auth failure from its message text.

The adapter and `GenerateReply` both need this answer and must not disagree, so
it lives in neither. There is no exception class to catch: `init_chat_model`
resolves whichever provider the deployment named, and each SDK raises its own
type. The markers are what every OpenAI-compatible API sends for this case.
"""

_AUTH_MARKERS = (
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "error code: 401",
    "unauthorized",
)


def is_auth_failure(detail: str) -> bool:
    """True when the provider rejected our credentials, not the request.

    A missing, wrong or revoked key fails the same way for every user until an
    operator fixes it. Retrying cannot help, and the message names the key and
    links to the vendor's dashboard, so it is not for a user either.
    """
    lowered = detail.lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)
