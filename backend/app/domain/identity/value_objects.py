"""Value objects for identity: what counts as an address, and what counts as a password."""

import re
from dataclasses import dataclass

from app.domain.identity.errors import InvalidEmail, WeakPassword

# Permissive on purpose. A strict regex rejects real addresses. This only
# rejects what clearly cannot receive mail.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

_MAX_EMAIL_LENGTH = 254
"""The SMTP maximum, and the column size."""

MIN_PASSWORD_LENGTH = 12
"""Minimum password length. A business rule, not a setting."""

MAX_PASSWORD_LENGTH = 256
"""Maximum password length, so a huge input cannot reach the hasher."""


@dataclass(frozen=True, slots=True)
class Email:
    """A normalised email address, and the identity a user is known by."""

    value: str

    @classmethod
    def sanitize(cls, raw: str | None) -> "Email":
        """Normalise an untrusted address, or refuse it. Lowercased here, once."""
        candidate = (raw or "").strip().lower()
        if not candidate or len(candidate) > _MAX_EMAIL_LENGTH:
            raise InvalidEmail(raw)
        if _EMAIL_PATTERN.match(candidate) is None:
            raise InvalidEmail(raw)
        return cls(candidate)


@dataclass(frozen=True, slots=True)
class RawPassword:
    """A password as typed, checked once here and never stored."""

    value: str

    def __post_init__(self) -> None:
        if len(self.value) < MIN_PASSWORD_LENGTH:
            raise WeakPassword(f"it must be at least {MIN_PASSWORD_LENGTH} characters")
        if len(self.value) > MAX_PASSWORD_LENGTH:
            raise WeakPassword(f"it must be at most {MAX_PASSWORD_LENGTH} characters")
        if self.value.strip() != self.value:
            # Usually a copy-paste accident that locks the user out later.

            raise WeakPassword("it must not start or end with whitespace")
