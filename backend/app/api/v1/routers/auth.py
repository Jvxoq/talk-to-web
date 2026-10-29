"""Authentication routes.

Splits a session into a JSON body and a refresh cookie, and reads the cookie
back. The use cases know nothing about cookies. Sign-out is a POST, because
CORS allows only GET, POST and OPTIONS.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Cookie, Response, status

from app.api.dependencies import (
    AuthenticateUserDep,
    ClientIpDep,
    CurrentUserDep,
    RefreshCookieDep,
    RefreshSessionDep,
    RegisterUserDep,
    RevokeSessionDep,
)
from app.api.v1.schemas.auth import Credentials, SessionOut, UserOut
from app.application.identity.dto import (
    LoginInput,
    RefreshCookiePolicy,
    RefreshInput,
    RegisterInput,
    SessionTokens,
)
from app.domain.identity.errors import InvalidToken

router = APIRouter(prefix="/auth", tags=["auth"])

# FastAPI needs a constant alias to read a cookie. The configured name is used
# only for writing, so a renamed cookie must be renamed here too.
_COOKIE_ALIAS = "refresh_token"

# Use with `= None`. Without a default, a missing cookie is a 422, not our 401.
RefreshCookie = Annotated[str | None, Cookie(alias=_COOKIE_ALIAS)]


def _attach(response: Response, tokens: SessionTokens, policy: RefreshCookiePolicy) -> SessionOut:
    """Put the refresh half in a cookie and return the half the client may read."""
    response.set_cookie(
        key=policy.name,
        value=tokens.refresh_token,
        max_age=tokens.refresh_expires_in,
        # Scripts cannot read it, so XSS gets only the short-lived access token.
        httponly=True,
        secure=policy.secure,
        samesite=_samesite(policy),
        path=policy.path,
        domain=policy.domain,
    )
    return SessionOut.from_domain(tokens)


def _samesite(policy: RefreshCookiePolicy) -> Literal["lax", "strict", "none"]:
    """Narrow the setting to Starlette's literal type. Unknown values become `lax`."""
    value = policy.samesite.lower()
    if value == "strict":
        return "strict"
    if value == "none":
        return "none"
    return "lax"


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: Credentials,
    response: Response,
    use_case: RegisterUserDep,
    policy: RefreshCookieDep,
    ip: ClientIpDep,
) -> SessionOut:
    tokens = await use_case(RegisterInput(email=body.email, password=body.password, client_ip=ip))
    return _attach(response, tokens, policy)


@router.post("/login")
async def login(
    body: Credentials,
    response: Response,
    use_case: AuthenticateUserDep,
    policy: RefreshCookieDep,
    ip: ClientIpDep,
) -> SessionOut:
    tokens = await use_case(LoginInput(email=body.email, password=body.password, client_ip=ip))
    return _attach(response, tokens, policy)


@router.post("/refresh")
async def refresh(
    response: Response,
    use_case: RefreshSessionDep,
    policy: RefreshCookieDep,
    ip: ClientIpDep,
    refresh_token: RefreshCookie = None,
) -> SessionOut:
    if not refresh_token:
        # Normal on a first visit, but still a 401.
        raise InvalidToken("no session cookie")

    tokens = await use_case(RefreshInput(refresh_token=refresh_token, client_ip=ip))
    return _attach(response, tokens, policy)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    use_case: RevokeSessionDep,
    policy: RefreshCookieDep,
    refresh_token: RefreshCookie = None,
) -> Response:
    if refresh_token:
        await use_case(refresh_token)

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    # Always cleared, with the same attributes it was set with, or the browser keeps it.
    response.delete_cookie(
        key=policy.name,
        path=policy.path,
        domain=policy.domain,
        httponly=True,
        secure=policy.secure,
        samesite=_samesite(policy),
    )
    return response


@router.get("/me")
async def me(user: CurrentUserDep) -> UserOut:
    """Who the access token belongs to, read from its claims without a database call."""

    return UserOut.from_context(user)
