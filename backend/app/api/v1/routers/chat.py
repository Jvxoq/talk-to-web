"""Chat generation route: parse the body, hand it to the use case, stream frames."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.dependencies import ChatModelsDep, CurrentUserDep, GenerateReplyDep
from app.api.v1.schemas.chat import TextModelRequest
from app.api.v1.sse import to_sse
from app.application.chat.dto import GenerateReplyInput

router = APIRouter(tags=["chat"])


@router.post("/generate/text/")
async def generate_text(
    body: TextModelRequest,
    use_case: GenerateReplyDep,
    models: ChatModelsDep,
    user: CurrentUserDep,
) -> StreamingResponse:
    request = body.validated_against(models)
    # Awaited first, so a rate-limit refusal is a 429, not a cut-off 200.
    events = await use_case(
        GenerateReplyInput(
            model=request.model,
            user_input=request.user_input,
            owner_id=user.user_id,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
        )
    )
    return StreamingResponse(
        to_sse(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Stops proxy buffering, which would deliver the stream in one lump.
            "X-Accel-Buffering": "no",
        },
    )
