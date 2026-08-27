"""Reading a provider's failure text well enough to decide what to do with it.

Two callers need the same answer and must not drift apart: the adapter, which
decides whether an attempt is worth retrying, and `GenerateReply`, which decides
what the user is told. Neither owns the question, so it lives here.

Matching on message text is crude, and it is what is available. `init_chat_model`
resolves whichever provider the deployment named, and each SDK raises its own
exception type, so there is no class to catch that stays true across providers.
The markers below are the shape every OpenAI-compatible API uses for the case.
"""

_AUTH_MARKERS = (
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "error code: 401",
    "unauthorized",
)


def is_auth_failure(detail: str) -> bool:
    """True when the provider rejected our credentials rather than the request.

    This is an operator mistake - a missing, wrong or revoked key - and it will
    fail identically for every user until someone fixes the deployment. That
    makes it both un-retryable and something no user should be shown, since the
    provider's own message names the key and links to its dashboard.
    """
    lowered = detail.lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)
