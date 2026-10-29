"""What a readiness check reports. The router decides the wire format."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    """One dependency's verdict: a name and a boolean.

    No error text, because `/ready` is public and errors can hold secrets.
    """

    name: str
    ready: bool


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    components: tuple[ComponentReadiness, ...]

    @property
    def ready(self) -> bool:
        """True only if every dependency answered."""

        return all(component.ready for component in self.components)
