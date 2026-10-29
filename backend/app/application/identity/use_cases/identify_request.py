"""Turn a bearer token into the person making the request."""

from app.application.identity.dto import UserContext
from app.application.identity.ports import AccessTokenCodec
from app.domain.identity.errors import InvalidToken


class IdentifyRequest:
    """Verify an access token and say who it belongs to. No database call.

    A deactivated account works until its short-lived token expires.
    """

    def __init__(self, access_tokens: AccessTokenCodec) -> None:
        self._access_tokens = access_tokens

    async def __call__(self, token: str) -> UserContext:
        if not token.strip():
            raise InvalidToken("no token supplied")

        claims = self._access_tokens.decode(token)
        return UserContext(user_id=claims.user_id, email=claims.email)
