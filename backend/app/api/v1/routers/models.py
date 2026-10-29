"""What this deployment will answer on. The frontend builds its picker from it."""

from fastapi import APIRouter

from app.api.dependencies import ChatModelsDep
from app.api.v1.schemas.chat import ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models/")
async def list_models(models: ChatModelsDep) -> ModelsResponse:
    # A plain passthrough of config, so no use case. The first model is the
    # default. `LangChainChatModel` guarantees the list is not empty.

    return ModelsResponse(models=list(models), default=models[0])
