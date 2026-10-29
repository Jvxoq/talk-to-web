"""Wire shapes for the chat endpoints."""

from collections.abc import Sequence
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.domain.chat.errors import UnsupportedModel

UserInput = Annotated[
    str,
    # Strip first, so a blank message is a 422 before the stream opens.
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32_000),
]


class TextModelRequest(BaseModel):
    # Allows the `model` field name without Pydantic's `model_` prefix warning.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Checked at request time against config. Length-bounded against abuse.
    model: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    user_input: UserInput
    temperature: float = Field(default=0.0, ge=0, le=2)
    # The agent's memory key. `None` is a one-off turn with no history.
    conversation_id: int | None = Field(default=None, ge=1)

    def validated_against(self, models: Sequence[str]) -> Self:
        """Reject a model this deployment does not serve.

        Not a field validator, because the list comes from the container. Runs
        before the stream opens, so it can still be a 400.
        """

        if self.model not in models:
            raise UnsupportedModel(self.model)
        return self


class ModelsResponse(BaseModel):
    """What the frontend populates its model picker from."""

    models: list[str]
    default: str
