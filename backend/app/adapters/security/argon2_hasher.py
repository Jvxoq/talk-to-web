"""Password hashing with Argon2id."""

from argon2 import PasswordHasher as Argon2
from argon2.exceptions import Argon2Error, InvalidHashError, VerifyMismatchError
from loguru import logger


class Argon2PasswordHasher:
    """Hashes and checks passwords. Implements `PasswordHasher`.

    Argon2id, not bcrypt, which truncates at 72 bytes. Library defaults, so the
    parameters follow current guidance on upgrade.
    """

    def __init__(self) -> None:
        self._hasher = Argon2()
        # Computed once, so unknown-email logins cost the same as known ones.
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
            # A corrupt hash looks like a wrong password, not a 500.
            # `InvalidHashError` is a `ValueError`, so it is listed separately.

            logger.warning(f"Could not verify a stored password hash: {error}")
            return False

    def dummy_hash(self) -> str:
        return self._dummy
