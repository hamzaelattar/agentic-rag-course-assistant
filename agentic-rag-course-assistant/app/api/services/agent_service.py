import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DATABASE_PATH = Path("data/conversations.sqlite3")


# The local GPU has 4 GB VRAM. Serialize complete graph invocations.
_agent_lock = threading.Lock()
_metadata_lock = threading.Lock()


def _connect_metadata():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS web_conversations ("
        "thread_id TEXT PRIMARY KEY, title TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def create_conversation(thread_id: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with _metadata_lock, _connect_metadata() as connection:
        connection.execute(
            "INSERT INTO web_conversations(thread_id, title, updated_at) VALUES (?, ?, ?)",
            (thread_id, "Nouvelle conversation", now),
        )
    return {"thread_id": thread_id, "title": "Nouvelle conversation", "updated_at": now}


def list_conversations() -> list[dict]:
    with _metadata_lock, _connect_metadata() as connection:
        rows = connection.execute(
            "SELECT thread_id, title, updated_at FROM web_conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {"thread_id": row[0], "title": row[1], "updated_at": row[2]}
        for row in rows
    ]


def load_conversation(thread_id: str) -> dict:
    with _metadata_lock, _connect_metadata() as connection:
        row = connection.execute(
            "SELECT thread_id, title, updated_at FROM web_conversations WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    if row is None:
        raise KeyError(thread_id)
    from app.agent.memory import get_conversation_history, open_conversation_agent

    with _agent_lock, open_conversation_agent() as agent:
        messages = get_conversation_history(agent, thread_id)
    return {"thread_id": row[0], "title": row[1], "updated_at": row[2], "messages": messages}


def send_message(thread_id: str, message: str) -> dict:
    with _metadata_lock, _connect_metadata() as connection:
        row = connection.execute(
            "SELECT title FROM web_conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    if row is None:
        raise KeyError(thread_id)

    from app.agent.memory import ask_agent, open_conversation_agent

    with _agent_lock, open_conversation_agent() as agent:
        result = ask_agent(agent, thread_id, message)

    title = message.strip().replace("\n", " ")[:72]
    now = datetime.now(timezone.utc).isoformat()
    with _metadata_lock, _connect_metadata() as connection:
        connection.execute(
            "UPDATE web_conversations SET title = ?, updated_at = ? WHERE thread_id = ?",
            (title, now, thread_id),
        )
    return {
        "thread_id": thread_id,
        "answer": result.get("answer", ""),
        "rewritten_question": result.get("rewritten_question", ""),
        "selected_tools": result.get("selected_tools", []),
        "sources": result.get("sources", []),
        "diagram_code": result.get("diagram_code"),
        "answer_valid": result.get("answer_valid"),
        "rewrite_attempts": result.get("rewrite_attempts", 0),
    }


def delete_conversation(thread_id: str) -> bool:
    # LangGraph checkpoints are stored in the same SQLite DB. Do not expose a
    # partial deletion until checkpoint and metadata removal can be atomic.
    return False
