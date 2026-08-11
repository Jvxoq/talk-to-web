"""Password hashing with Argon2id."""

from argon2 import PasswordHasher as Argon2
from argon2.exceptions import Argon2Error, InvalidHashError, VerifyMismatchError
from loguru import logger


class Argon2PasswordHasher:
    """
    Hashes and checks passwords, satisfying `app.application.identity.ports.PasswordHasher`.

    Argon2id rather than bcrypt: bcrypt silently truncates at 72 bytes, so a long
    passphrase quietly becomes a shorter one and every password sharing its first
    72 bytes unlocks the account. Argon2 has no such edge, and it is memory-hard,
    which is what makes GPU-scale guessing expensive rather than merely slow.

    The parameters are the library's defaults on purpose. They track current
    guidance as the library is upgraded; a set pinned here would be frozen at
    whatever was reasonable the day it was written.
    """

    def __init__(self) -> None:
        self._hasher = Argon2()
        # Computed once at construction, not per failed login: it exists to make
        # a login for an unknown address cost the same as one for a known
        # address, and hashing a fresh dummy every time would cost strictly more.
        self._dummy = self._hasher.hash("a password that belongs to nobody")

    def hash(self, plain: str) -> str:
        return self._hasher.hash(plain)

    def verify(self, plain: str, hashed: str) -> bool:
        """Whether `plain` produced `hashed`. Never raises."""
        try:
            return self._hasher.verify(hashed, plain)
        except VerifyMismatchError:
            return False
        except (Argon2Error, InvalidHashError) as error:
            # A corrupt or foreign hash string. Reported as a failed check rather
            # than a 500: the caller cannot fix it, and it must not be
            # distinguishable from a wrong password.
            #
            # `InvalidHashError` is named separately because it descends from
            # `ValueError`, not from `Argon2Error` - catching only the latter
            # let an unparseable stored hash escape as an unhandled exception.
            logger.warning(f"Could not verify a stored password hash: {error}")
            return False

    def dummy_hash(self) -> str:
        return self._dummy
