"""Wire shapes for signing up, signing in and staying signed in."""

from pydantic import BaseModel, ConfigDict, Field

from app.application.identity.dto import SessionTokens, UserContext
from app.domain.identity.value_objects import MAX_PASSWORD_LENGTH

# The bounds here are not the password policy - `RawPassword` in the domain owns
# that, and it is what returns a readable "too short" to the user. These are the
# edge's own job: refusing a megabyte string before it reaches an Argon2 call.
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
    """What a successful sign-in returns.

    The refresh token is absent on purpose. It goes into an httpOnly cookie the
    page's JavaScript cannot read, which is the entire reason for splitting the
    pair: a script that manages to run on this origin can steal the access token
    and hold it for its remaining minutes, not the session for a fortnight.
    """

    access_token: str
    # Not a secret, despite the name: it is the RFC 6750 scheme the client must
    # put in front of the token, and it is the same string for every response.
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
