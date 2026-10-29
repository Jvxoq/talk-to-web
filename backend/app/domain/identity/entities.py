"""Identity entities: who holds an account, and which sessions are still alive."""

from dataclasses import dataclass
from datetime import datetime

from app.domain.identity.value_objects import Email


@dataclass(slots=True)
class User:
    """Someone with an account, and the owner every conversation hangs off."""

    email: Email
    password_hash: str
    id: int | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def is_persisted(self) -> bool:
        return self.id is not None


@dataclass(slots=True)
class RefreshToken:
    """One long-lived session, stored as a one-way fingerprint, never the secret."""

    user_id: int
    fingerprint: str
    expires_at: datetime
    id: int | None = None
    revoked_at: datetime | None = None
    created_at: datetime | None = None

    def is_usable(self, now: datetime) -> bool:
        """Whether this session may still be exchanged for a new pair."""
        return self.revoked_at is None and now < self.expires_at

    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def revoke(self, now: datetime) -> None:
        """End this session. Idempotent, so the first revoke time is kept."""

        if self.revoked_at is None:
            self.revoked_at = now
