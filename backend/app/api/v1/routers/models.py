"""What this deployment will answer on. The frontend builds its picker from it."""

from fastapi import APIRouter

from app.api.dependencies import ChatModelsDep
from app.api.v1.schemas.chat import ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/models/")
async def list_models(models: ChatModelsDep) -> ModelsResponse:
    # A sanctioned passthrough: read configuration, serialize it. No policy, no
    # second collaborator, nothing a use case would add but a file.
    #
    # First-configured is the default by convention - the operator orders
    # `LLM_MODELS` and the top of that list is what a new chat opens with.
    # Indexing is safe because an empty list never reaches a running app:
    # `LangChainChatModel` refuses to construct without at least one model, and
    # it is built before the container is.
    return ModelsResponse(models=list(models), default=models[0])
