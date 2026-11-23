import asyncio

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.api.services.agent_service import send_message

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    try:
        return await asyncio.to_thread(
            send_message, payload.thread_id, payload.message
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation introuvable.") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"L’agent n’a pas pu traiter la demande : {type(exc).__name__}.",
        ) from exc
