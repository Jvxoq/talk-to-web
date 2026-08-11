"""Business failures in the identity context. No HTTP status codes live here."""


class IdentityError(Exception):
    """Base class for every failure the identity context can name."""


class InvalidEmail(IdentityError):
    def __init__(self, raw: str | None) -> None:
        self.raw = raw
        super().__init__(f"{raw!r} is not a usable email address")


class WeakPassword(IdentityError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Password rejected: {reason}")


class EmailAlreadyRegistered(IdentityError):
    def __init__(self, email: str) -> None:
        self.email = email
        super().__init__(f"{email} is already registered")


class InvalidCredentials(IdentityError):
    """Wrong email, wrong password, or a disabled account.

    Deliberately one error for all three. Telling a caller which half was wrong
    turns the login form into a way to discover who has an account here.
    """

    def __init__(self) -> None:
        super().__init__("Email or password is incorrect")


class InvalidToken(IdentityError):
    def __init__(self, reason: str = "malformed or unrecognised") -> None:
        self.reason = reason
        super().__init__(f"Token rejected: {reason}")


class TokenExpired(IdentityError):
    def __init__(self) -> None:
        super().__init__("Token has expired")
