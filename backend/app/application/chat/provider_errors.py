"""Recognises a provider's auth failure from its message text.

Shared by the adapter and `GenerateReply`. Each SDK raises its own type, so
this matches common OpenAI-compatible message text instead.
"""

_AUTH_MARKERS = (
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "error code: 401",
    "unauthorized",
)


def is_auth_failure(detail: str) -> bool:
    """True when the provider rejected our key. Retrying cannot help."""

    lowered = detail.lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)
