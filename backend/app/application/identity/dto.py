"""Inputs and outputs owned by the identity use cases — not the wire, not the database."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RegisterInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    password: str
    # Rate-limit key only. `None` when untrusted, so a proxy IP never limits everyone.
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
    """A verified token's claims. Includes the email, so no database call is needed."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    email: str


class UserContext(BaseModel):
    """Who is making this request. The only identity the rest of the app sees."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    email: str


class SessionTokens(BaseModel):
    """A new token pair and its user.

    The raw refresh secret goes only into the cookie. The database gets its fingerprint.
    """

    model_config = ConfigDict(frozen=True)

    access_token: str
    expires_in: int
    refresh_token: str
    refresh_expires_in: int
    user: UserContext


class RefreshCookiePolicy(BaseModel):
    """How the refresh cookie is written. Reaches the API through the container."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    secure: bool
    samesite: str
    domain: str | None = None
