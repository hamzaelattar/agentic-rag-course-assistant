import asyncio
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import ConversationCreateResponse, ConversationDetail, ConversationSummary
from app.api.services.agent_service import (
    create_conversation,
    delete_conversation,
    list_conversations,
    load_conversation,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationCreateResponse)
async def create():
    thread_id = str(uuid4())
    await asyncio.to_thread(create_conversation, thread_id)
    return {"thread_id": thread_id}


@router.get("", response_model=list[ConversationSummary])
async def list_all():
    return await asyncio.to_thread(list_conversations)


@router.get("/{thread_id}", response_model=ConversationDetail)
async def get_one(thread_id: str):
    try:
        return await asyncio.to_thread(load_conversation, thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation introuvable.") from exc


@router.delete("/{thread_id}")
async def delete_one(thread_id: str):
    if not await asyncio.to_thread(delete_conversation, thread_id):
        raise HTTPException(
            status_code=501,
            detail="La suppression sûre des checkpoints LangGraph n’est pas configurée.",
        )
