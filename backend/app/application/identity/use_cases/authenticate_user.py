"""Sign someone in."""

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.dto import LoginInput, SessionTokens
from app.application.identity.ports import PasswordHasher, RateLimiter
from app.application.identity.sessions import SessionMinter
from app.domain.identity.errors import InvalidCredentials, InvalidEmail
from app.domain.identity.value_objects import Email


class AuthenticateUser:
    """
    Exchange an email and password for a session.

    Every failure is `InvalidCredentials`, whatever actually went wrong. A
    response that distinguishes "no such account" from "wrong password" is a way
    to test whether an address has signed up here, and the login form is
    reachable by anyone.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        hasher: PasswordHasher,
        sessions: SessionMinter,
        limiter: RateLimiter,
    ) -> None:
        self._uow_factory = uow_factory
        self._hasher = hasher
        self._sessions = sessions
        self._limiter = limiter

    async def __call__(self, data: LoginInput) -> SessionTokens:
        try:
            email = Email.sanitize(data.email)
        except InvalidEmail:
            # A malformed address is a failed login, not a validation error:
            # answering differently would tell a caller which addresses are even
            # worth guessing, before any rate limit has been spent.
            raise InvalidCredentials() from None

        # Counted before the password is checked, so a spray of guesses runs out
        # of budget rather than out of passwords. Keyed on the address first -
        # that is the thing being attacked - with the caller's IP as a second
        # ceiling when the deployment can be trusted to report it.
        await self._limiter.hit(f"login:{email.value}")
        if data.client_ip is not None:
            await self._limiter.hit(f"login-ip:{data.client_ip}")

        async with self._uow_factory() as uow:
            user = await uow.users.get_by_email(email)

            # Verify even when there is no user, against a hash of nothing. The
            # work is the point: returning early here makes an unknown address
            # answer measurably faster than a known one.
            expected = user.password_hash if user is not None else self._hasher.dummy_hash()
            matched = self._hasher.verify(data.password, expected)

            if user is None or not matched or not user.is_active:
                logger.info(f"Failed login for {email.value}")
                raise InvalidCredentials()

            tokens = await self._sessions.mint(uow, user)
            await uow.commit()

        await self._limiter.reset(f"login:{email.value}")
        logger.info(f"User {user.id} signed in")
        return tokens
