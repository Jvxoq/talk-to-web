"""Sign someone in."""

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.dto import LoginInput, SessionTokens
from app.application.identity.ports import PasswordHasher, RateLimiter
from app.application.identity.sessions import SessionMinter
from app.domain.identity.errors import InvalidCredentials, InvalidEmail
from app.domain.identity.value_objects import Email


class AuthenticateUser:
    """Exchange an email and password for a session.

    Every failure is `InvalidCredentials`, so login cannot reveal who has an account.
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
            # Same answer as a wrong password, before any limit is spent.
            raise InvalidCredentials() from None

        # Counted before checking. Keyed on the email, plus the IP when trusted.
        await self._limiter.hit(f"login:{email.value}")
        if data.client_ip is not None:
            await self._limiter.hit(f"login-ip:{data.client_ip}")

        async with self._uow_factory() as uow:
            user = await uow.users.get_by_email(email)

            # Always verify, so unknown emails take the same time as known ones.

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
