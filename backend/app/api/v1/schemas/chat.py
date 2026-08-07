"""Wire shapes for the chat endpoints."""

from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.domain.chat.errors import UnsupportedModel

UserInput = Annotated[
    str,
    # Stripping before the length check is what makes a whitespace-only message
    # a 422 here rather than a domain error raised after the SSE response has
    # already started, where nothing can turn it back into a status code.
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32_000),
]


class TextModelRequest(BaseModel):
    # `model` is a frozen field name in the frontend contract, but Pydantic
    # reserves the `model_` prefix for its own API (`model_dump`, `model_config`)
    # and warns about any field that shadows it. Clearing protected_namespaces
    # keeps the wire name without the warning.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # No longer a `Literal`: the model list is deployment configuration now, so
    # it cannot be baked into a type at import time. Bounded anyway, because an
    # unbounded string is a field an attacker can send megabytes into.
    model: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    user_input: UserInput
    temperature: float = Field(default=0.0, ge=0, le=2)
    # The agent's memory key. `None` is a one-off turn with no history.
    conversation_id: int | None = Field(default=None, ge=1)

    def validated_against(self, models: Sequence[str]) -> Self:
        """Reject a model this deployment does not serve.

        The check cannot be a field validator: the allowed list lives in settings,
        which the API layer is forbidden to import, so it arrives from the
        container at request time instead. Raising the domain error means the
        existing error map turns it into a 400 - before the stream opens, where a
        status code still means something. Left to reach the adapter, the same
        mistake would surface mid-body as an error frame inside a 200.
        """
        if self.model not in models:
            raise UnsupportedModel(self.model)
        return self


class ModelsResponse(BaseModel):
    """What the frontend populates its model picker from."""

    models: list[str]
    default: str
