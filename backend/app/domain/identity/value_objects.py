"""Value objects for identity: what counts as an address, and what counts as a password."""

import re
from dataclasses import dataclass

from app.domain.identity.errors import InvalidEmail, WeakPassword

# Deliberately permissive. RFC 5321 allows addresses that no regex reads
# correctly, and a strict pattern's only measurable effect is turning away real
# people with unusual-but-valid addresses. Confirming an address is a job for a
# confirmation email; this only rejects what obviously cannot be delivered to.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

_MAX_EMAIL_LENGTH = 254
"""The longest address SMTP will carry, and the ceiling the column is sized to."""

MIN_PASSWORD_LENGTH = 12
"""How short a password may be.

A business rule, not configuration: it belongs with the other rules about who
may hold an account, and a deployment that could lower it by setting an
environment variable would eventually be a deployment that had.
"""

MAX_PASSWORD_LENGTH = 256
"""A ceiling, so a megabyte 'password' cannot be handed to the hasher.

Argon2 costs memory and CPU proportional to nothing the caller controls except
the input length; without a bound, the login endpoint hashes whatever it is
sent.
"""


@dataclass(frozen=True, slots=True)
class Email:
    """A normalised email address, and the identity a user is known by."""

    value: str

    @classmethod
    def sanitize(cls, raw: str | None) -> "Email":
        """Normalise an untrusted address, or refuse it.

        Case folding happens here rather than at the database, because two
        people typing the same address in different cases are one person, and
        which layer decides that must not depend on which query ran.
        """
        candidate = (raw or "").strip().lower()
        if not candidate or len(candidate) > _MAX_EMAIL_LENGTH:
            raise InvalidEmail(raw)
        if _EMAIL_PATTERN.match(candidate) is None:
            raise InvalidEmail(raw)
        return cls(candidate)


@dataclass(frozen=True, slots=True)
class RawPassword:
    """A password as the user typed it, checked but never stored.

    Exists so that "is this acceptable?" is answered once, in the domain, rather
    than in whichever route happens to accept a password next. Only the hash of
    this ever leaves the use case.
    """

    value: str

    def __post_init__(self) -> None:
        if len(self.value) < MIN_PASSWORD_LENGTH:
            raise WeakPassword(f"it must be at least {MIN_PASSWORD_LENGTH} characters")
        if len(self.value) > MAX_PASSWORD_LENGTH:
            raise WeakPassword(f"it must be at most {MAX_PASSWORD_LENGTH} characters")
        if self.value.strip() != self.value:
            # Leading or trailing whitespace is almost always a copy-paste
            # accident, and it locks the account out on the next manual typing.
            raise WeakPassword("it must not start or end with whitespace")
