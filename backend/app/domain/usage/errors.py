"""Business failures about how much of this service one caller may consume.

Its own context rather than a corner of identity, because the answer to "who
are you?" and the answer to "how often may you ask?" are different questions.
Signing in is rate limited to stop guessing; chatting and uploading are rate
limited because every call spends money at a model provider. One error type
serves both - the caller is told the same thing either way - and no HTTP status
code lives here.
"""


class UsageError(Exception):
    """Base class for every failure the usage context can name."""


class RateLimited(UsageError):
    """The budget for this key is spent, and when it will not be.

    `retry_after_seconds` is part of the failure, not a hint: a caller told to
    back off without being told for how long backs off for no time at all.
    """

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Too many requests. Try again in {retry_after_seconds} seconds")
