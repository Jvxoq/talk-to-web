"""Provider failures the adapter and `GenerateReply` must agree on.

They live in neither, because both need the same answer. An auth failure is
recognised from its message text: there is no exception class to catch, since
`init_chat_model` resolves whichever provider the deployment named and each SDK
raises its own type. The markers are what every OpenAI-compatible API sends for
this case. A timeout is recognised by type in the adapter, which raises
`ProviderTimeoutError`.
"""


class ProviderTimeoutError(Exception):
    """The model provider did not answer within the per-call timeout.

    A type, so the use case never guesses from text: a provider's 400 body that
    quotes a `timeout` parameter is not a timeout.
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
