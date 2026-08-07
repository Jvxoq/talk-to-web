"""Conversation routes. Each body is parse -> call use case -> serialize."""

from fastapi import APIRouter, Response, status

from app.api.dependencies import (
    DeleteConversationDep,
    GetConversationDep,
    RecordExchangeDep,
    StartConversationDep,
)
from app.api.v1.schemas.conversations import (
    ConversationCreate,
    ConversationDetailOut,
    ConversationOut,
    MessageCreate,
    MessageOut,
)
from app.application.chat.dto import RecordExchangeInput, StartConversationInput

router = APIRouter(tags=["conversations"])


@router.post("/conversations/", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate,
    use_case: StartConversationDep,
) -> ConversationOut:
    conversation = await use_case(
        StartConversationInput(title=body.title, model_type=body.model_type)
    )
    return ConversationOut.from_domain(conversation)


@router.get("/conversations/{conversation_id}")
async def read_conversation(
    conversation_id: int,
    use_case: GetConversationDep,
) -> ConversationDetailOut:
    conversation = await use_case(conversation_id)
    return ConversationDetailOut.from_domain(conversation)


@router.post("/conversations/{conversation_id}/messages", status_code=status.HTTP_201_CREATED)
async def create_message(
    conversation_id: int,
    body: MessageCreate,
    use_case: RecordExchangeDep,
) -> MessageOut:
    message = await use_case(
        RecordExchangeInput(
            conversation_id=conversation_id,
            prompt_content=body.prompt_content,
            response_content=body.response_content,
            prompt_tokens=body.prompt_tokens,
            response_tokens=body.response_tokens,
            total_tokens=body.total_tokens,
            is_success=body.is_success,
            status_code=body.status_code,
        )
    )
    return MessageOut.from_domain(message)


# POST rather than DELETE on purpose: the browser fires this from
# `navigator.sendBeacon` as the tab unloads, and sendBeacon can only send POST.
@router.post("/conversations/{conversation_id}/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    use_case: DeleteConversationDep,
) -> Response:
    await use_case(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
