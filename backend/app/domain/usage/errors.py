"""Business failures about how much one caller may use.

Separate from identity: "who are you" and "how often" are different questions.
"""


class UsageError(Exception):
    """Base class for every failure the usage context can name."""


class RateLimited(UsageError):
    """The budget for this key is spent. `retry_after_seconds` says for how long."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Too many requests. Try again in {retry_after_seconds} seconds")
