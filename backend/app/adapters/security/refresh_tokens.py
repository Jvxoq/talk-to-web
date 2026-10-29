"""Refresh secrets, and the fingerprints that stand in for them in the database."""

import hashlib
import secrets

_SECRET_BYTES = 32
"""256 bits of entropy, too many to guess."""


class Sha256RefreshTokenFactory:
    """Mints refresh secrets and fingerprints them. Implements `RefreshTokenFactory`.

    SHA-256, not Argon2: random 256-bit secrets need no slow hash. Opaque, not
    a JWT, so a session can be revoked.
    """

    def new_secret(self) -> str:
        return secrets.token_urlsafe(_SECRET_BYTES)

    def fingerprint(self, secret: str) -> str:
        """The stored stand-in for a secret. 64 hex characters, always."""
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()
