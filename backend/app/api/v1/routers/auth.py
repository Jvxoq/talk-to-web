"""Authentication routes.

Two things live here that live nowhere else in the API: the split of a minted
session across a JSON body and a Set-Cookie header, and the reading of that
cookie back. Both are delivery decisions - the use cases return one
`SessionTokens` and have no idea either mechanism exists.

Every route is a POST, including sign-out. `app.factory` allows GET, POST and
OPTIONS across origins, so a `DELETE /auth/session` would be blocked by CORS
before it ever reached this module.
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

# Named so the parameter can be `refresh_token` while the cookie keeps whatever
# name the deployment configured. FastAPI reads a `Cookie()` parameter by its
# alias, which has to be a constant - it cannot be resolved from the container
# per request - so the configurable name is only used when *writing* the cookie.
# A deployment that renames it must rename it here too; the default is the one
# both sides agree on.
_COOKIE_ALIAS = "refresh_token"

# The `= None` at each use site is load-bearing: an `Annotated[... , Cookie()]`
# parameter with no default is *required*, so a caller with no cookie would get
# a 422 from FastAPI's validation instead of the 401 this app means. Arriving
# without one is the ordinary state of a first visit.
RefreshCookie = Annotated[str | None, Cookie(alias=_COOKIE_ALIAS)]


def _attach(response: Response, tokens: SessionTokens, policy: RefreshCookiePolicy) -> SessionOut:
    """Put the refresh half in a cookie and return the half the client may read."""
    response.set_cookie(
        key=policy.name,
        value=tokens.refresh_token,
        max_age=tokens.refresh_expires_in,
        # The whole point: unreadable to any script on the page, so an XSS gets
        # at most a short-lived access token rather than a fortnight of session.
        httponly=True,
        secure=policy.secure,
        samesite=_samesite(policy),
        path=policy.path,
        domain=policy.domain,
    )
    return SessionOut.from_domain(tokens)


def _samesite(policy: RefreshCookiePolicy) -> Literal["lax", "strict", "none"]:
    """Narrow the configured value to what Starlette's signature accepts.

    An unrecognised setting falls back to `lax` rather than raising: a typo in an
    environment variable should not take a deployment's sign-in down, and `lax`
    is the strictest of the three that still works same-origin.
    """
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
        # No cookie at all is the ordinary state of a first visit, not an
        # anomaly. It is still a 401, because the answer to "am I signed in?"
        # is no.
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
    # Cleared whether or not there was anything to revoke, and with the same
    # attributes it was written with: a browser matches a deletion to a cookie
    # by name, path and domain, so a mismatch here leaves it in place and the
    # user "signed out" into a session that still works.
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
    """Who the presented access token belongs to.

    Answered from the token's own claims, with no database round trip - which
    also makes this the cheapest way for a client to find out whether the token
    it is holding is still good.
    """
    return UserOut.from_context(user)
