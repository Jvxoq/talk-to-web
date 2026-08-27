"""`is_auth_failure` - the string check both the adapter and the use case trust."""

import pytest

from app.application.chat.provider_errors import is_auth_failure

TOGETHER_401 = (
    "Error code: 401 - {'id': 'ad99e814-aws_ec1', 'error': {'message': 'Invalid API key "
    "provided. You can find your API key at https://api.together.ai/settings/api-keys.', "
    "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
)


@pytest.mark.parametrize(
    "detail",
    [
        TOGETHER_401,
        "Error code: 401 - Unauthorized",
        "openai.AuthenticationError: authentication_error",
        "INVALID_API_KEY",
    ],
)
def test_a_rejected_credential_is_recognised(detail: str) -> None:
    assert is_auth_failure(detail)


@pytest.mark.parametrize(
    "detail",
    [
        "rate_limit_exceeded",
        "Error code: 500 - upstream is down",
        "Connection reset by peer",
        # A token count is not a status code. The rate-limit branch matches
        # bare "401"-like numbers; this one must not.
        "prompt used 401 tokens",
        "",
    ],
)
def test_everything_else_is_left_alone(detail: str) -> None:
    assert not is_auth_failure(detail)
