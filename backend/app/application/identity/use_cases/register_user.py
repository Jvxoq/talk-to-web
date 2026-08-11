"""Open an account, and sign the new person in."""

from loguru import logger

from app.application.common.uow import UnitOfWorkFactory
from app.application.identity.dto import RegisterInput, SessionTokens
from app.application.identity.ports import PasswordHasher, RateLimiter
from app.application.identity.sessions import SessionMinter
from app.domain.identity.entities import User
from app.domain.identity.errors import EmailAlreadyRegistered
from app.domain.identity.value_objects import Email, RawPassword


class RegisterUser:
    """
    Create a user and hand back their first session.

    Registration is rate limited as well as login: without it the endpoint is a
    free way to burn Argon2 memory on every request, and a way to discover which
    addresses are taken one 409 at a time.
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

    async def __call__(self, data: RegisterInput) -> SessionTokens:
        if data.client_ip is not None:
            await self._limiter.hit(f"register:{data.client_ip}")

        # Both raise before anything is written or hashed.
        email = Email.sanitize(data.email)
        password = RawPassword(data.password)

        async with self._uow_factory() as uow:
            if await uow.users.get_by_email(email) is not None:
                # The unique index is the real guarantee - two simultaneous
                # registrations both pass this check - but losing that race
                # should be rare, not routine.
                raise EmailAlreadyRegistered(email.value)

            user = await uow.users.add(
                User(email=email, password_hash=self._hasher.hash(password.value))
            )
            tokens = await self._sessions.mint(uow, user)
            await uow.commit()

        logger.info(f"Registered user {user.id}")
        return tokens
