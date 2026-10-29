"""Transcription domain types: what a speech-to-text provider tells us."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transcript:
    """One transcript from the provider.

    `is_final=False` is an interim guess to overwrite. `speech_final` marks a pause.
    """

    text: str
    is_final: bool
    speech_final: bool = False

    def is_worth_sending(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """The shape of the PCM the browser is sending."""

    sample_rate: int
    channels: int = 1
    encoding: str = "linear16"

    def __post_init__(self) -> None:
        if not 8_000 <= self.sample_rate <= 192_000:
            raise ValueError(f"Implausible sample rate: {self.sample_rate}")
