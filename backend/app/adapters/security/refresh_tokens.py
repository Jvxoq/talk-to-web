"""Refresh secrets, and the fingerprints that stand in for them in the database."""

import hashlib
import secrets

_SECRET_BYTES = 32
"""256 bits of entropy. Guessing one is not a threat model at this size."""


class Sha256RefreshTokenFactory:
    """
    Mints refresh secrets and fingerprints them, satisfying
    `app.application.identity.ports.RefreshTokenFactory`.

    Plain SHA-256, not Argon2, and that is not an oversight. A password hash is
    slow because passwords are low-entropy and guessable; these secrets are 256
    random bits, so there is nothing to guess and a slow hash would only add
    latency to every refresh. What the digest buys is that the database never
    holds anything presentable: a leaked dump lists sessions, not credentials.

    Opaque random rather than a second JWT, because a session that cannot be
    revoked is not a session. This one is only meaningful next to its row.
    """

    def new_secret(self) -> str:
        return secrets.token_urlsafe(_SECRET_BYTES)

    def fingerprint(self, secret: str) -> str:
        """The stored stand-in for a secret. 64 hex characters, always."""
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()
