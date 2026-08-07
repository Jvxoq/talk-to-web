"""Business failures in the transcription context."""


class TranscriptionError(Exception):
    """Base class for every failure the transcription context can name."""


class MalformedStartFrame(TranscriptionError):
    def __init__(self, detail: str = "Malformed start frame") -> None:
        super().__init__(detail)


class TranscriptionUnavailable(TranscriptionError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
