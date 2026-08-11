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
    # Awaited, not just called: the use case spends the caller's rate-limit
    # budget before it returns the stream, so a refusal reaches the error
    # handler as a 429 instead of truncating a response that already sent 200.
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
            # nginx buffers proxied responses by default, which holds every SSE
            # frame until the buffer fills — the stream then arrives in one lump
            # long after the model finished, which reads to users as a hang.
            "X-Accel-Buffering": "no",
        },
    )
