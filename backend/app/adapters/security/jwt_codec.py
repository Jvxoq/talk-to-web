"""Signed access tokens, using JWT."""

from datetime import datetime, timedelta

import jwt
from loguru import logger

from app.application.identity.dto import IssuedAccessToken, TokenClaims
from app.domain.identity.errors import InvalidToken, TokenExpired


class JwtAccessTokenCodec:
    """
    Issues and verifies access tokens, satisfying
    `app.application.identity.ports.AccessTokenCodec`.

    A JWT because verification must not touch the database - this runs on every
    authenticated request. The price is that an issued token cannot be withdrawn
    before it expires, which is why the lifetime is minutes and why revocation
    lives on the refresh token instead.
    """

    def __init__(self, secret: str, algorithm: str, ttl_seconds: int) -> None:
        self._secret = secret
        self._algorithm = algorithm
        self._ttl = timedelta(seconds=ttl_seconds)

    def issue(self, user_id: int, email: str, now: datetime) -> IssuedAccessToken:
        expires_at = now + self._ttl
        token = jwt.encode(
            {
                # `sub` is a string by spec, and PyJWT enforces it on decode.
                "sub": str(user_id),
                "email": email,
                "iat": int(now.timestamp()),
                "exp": int(expires_at.timestamp()),
            },
            self._secret,
            algorithm=self._algorithm,
        )
        return IssuedAccessToken(token=token, expires_at=expires_at)

    def decode(self, token: str) -> TokenClaims:
        """Verify a token and read it, or raise a domain error."""
        try:
            payload = jwt.decode(
                token,
                self._secret,
                # A list of exactly the algorithm we issue with. Passing the
                # token's own `alg` header back here is the classic JWT hole:
                # "none" verifies anything, and HMAC-vs-RSA confusion lets a
                # public key be used as a signing secret.
                algorithms=[self._algorithm],
                options={"require": ["exp", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise TokenExpired() from None
        except jwt.PyJWTError as error:
            # Includes a bad signature, a wrong algorithm and a missing claim.
            # The reason is logged, never returned - it would tell a caller which
            # part of their forgery to fix next.
            logger.debug(f"Rejected an access token: {error}")
            raise InvalidToken() from None

        try:
            user_id = int(payload["sub"])
            email = str(payload["email"])
        except (KeyError, TypeError, ValueError):
            # A correctly signed token that is not one of ours - a stale format
            # after a claim change, most likely.
            raise InvalidToken("unrecognised claims") from None

        return TokenClaims(user_id=user_id, email=email)
