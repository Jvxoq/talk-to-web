"""Inputs and outputs owned by the identity use cases — not the wire, not the database."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RegisterInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    password: str
    # Only ever used as a rate-limit key. It is `None` whenever the deployment
    # cannot vouch for it — behind a proxy that does not set forwarded headers,
    # the address the app sees is the proxy's, and limiting on that would
    # throttle everyone at once.
    client_ip: str | None = None


class LoginInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    password: str
    client_ip: str | None = None


class RefreshInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    refresh_token: str
    client_ip: str | None = None


class IssuedAccessToken(BaseModel):
    """A signed access token and the moment it stops being accepted."""

    model_config = ConfigDict(frozen=True)

    token: str
    expires_at: datetime


class TokenClaims(BaseModel):
    """What a verified access token asserts.

    Carries the email as well as the id so that identifying a request costs no
    database round trip — this runs on every authenticated call.
    """

    model_config = ConfigDict(frozen=True)

    user_id: int
    email: str


class UserContext(BaseModel):
    """Who is making this request. The only identity the rest of the app sees."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    email: str


class SessionTokens(BaseModel):
    """One freshly minted pair, plus the user they belong to.

    The refresh token is the raw secret and appears exactly here: the API puts
    it straight into a cookie, and only its fingerprint reaches the database.
    """

    model_config = ConfigDict(frozen=True)

    access_token: str
    expires_in: int
    refresh_token: str
    refresh_expires_in: int
    user: UserContext


class RefreshCookiePolicy(BaseModel):
    """How the refresh cookie is written, decided by configuration.

    Lives in the application layer for the same reason `chat_models` does: the
    API needs it and may not read settings, so it arrives through the container.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    secure: bool
    samesite: str
    domain: str | None = None
