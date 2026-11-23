import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router

app = FastAPI(title="Agentic RAG Course Assistant API", version="1.0.0")
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
allowed_origins = list(dict.fromkeys([
    frontend_origin,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "agentic-rag-course-assistant"}
