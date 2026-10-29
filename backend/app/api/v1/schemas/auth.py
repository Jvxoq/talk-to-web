"""Wire shapes for signing up, signing in and staying signed in."""

from pydantic import BaseModel, ConfigDict, Field

from app.application.identity.dto import SessionTokens, UserContext
from app.domain.identity.value_objects import MAX_PASSWORD_LENGTH

# Not the password policy (`RawPassword` owns that). Just stops huge strings early.
_PASSWORD_FIELD = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=1, max_length=254)
    password: str = _PASSWORD_FIELD


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str

    @classmethod
    def from_context(cls, user: UserContext) -> "UserOut":
        return cls(id=user.user_id, email=user.email)


class SessionOut(BaseModel):
    """What a sign-in returns. The refresh token goes in an httpOnly cookie instead."""

    access_token: str
    # The RFC 6750 scheme name, not a secret.

    token_type: str = "bearer"  # noqa: S105
    expires_in: int
    user: UserOut

    @classmethod
    def from_domain(cls, tokens: SessionTokens) -> "SessionOut":
        return cls(
            access_token=tokens.access_token,
            expires_in=tokens.expires_in,
            user=UserOut.from_context(tokens.user),
        )
