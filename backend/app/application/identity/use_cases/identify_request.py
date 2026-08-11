"""Turn a bearer token into the person making the request."""

from app.application.identity.dto import UserContext
from app.application.identity.ports import AccessTokenCodec
from app.domain.identity.errors import InvalidToken


class IdentifyRequest:
    """
    Verify an access token and say who it belongs to.

    No database round trip, by design: this runs on every authenticated request,
    and the signature plus the expiry is the whole check. The cost is that a
    deactivated account keeps working until its access token expires, which is
    what bounds the access TTL to minutes and puts revocation on the refresh
    token instead.
    """

    def __init__(self, access_tokens: AccessTokenCodec) -> None:
        self._access_tokens = access_tokens

    async def __call__(self, token: str) -> UserContext:
        if not token.strip():
            raise InvalidToken("no token supplied")

        claims = self._access_tokens.decode(token)
        return UserContext(user_id=claims.user_id, email=claims.email)
