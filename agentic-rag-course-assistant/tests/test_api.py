from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import chat, conversations


client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_and_list_conversation(monkeypatch):
    monkeypatch.setattr(
        conversations, "create_conversation", lambda thread_id: {"thread_id": thread_id}
    )
    monkeypatch.setattr(conversations, "list_conversations", lambda: [])
    created = client.post("/api/conversations")
    assert created.status_code == 200
    assert len(created.json()["thread_id"]) == 36
    assert client.get("/api/conversations").json() == []


def test_chat_returns_agent_contract(monkeypatch):
    result = {
        "thread_id": "conversation_001", "answer": "Réponse du graphe",
        "rewritten_question": "Question reformulée", "selected_tools": ["course_rag"],
        "sources": [{"source": "cours.pdf", "page": 3}],
        "diagram_code": None, "answer_valid": True, "rewrite_attempts": 0,
    }
    monkeypatch.setattr(chat, "send_message", lambda thread_id, message: result)
    response = client.post(
        "/api/chat", json={"thread_id": "conversation_001", "message": "Explique"}
    )
    assert response.status_code == 200
    assert response.json() == result


def test_chat_validation_and_missing_thread(monkeypatch):
    assert client.post("/api/chat", json={"thread_id": "bad/id", "message": "x"}).status_code == 422
    monkeypatch.setattr(chat, "send_message", lambda *_: (_ for _ in ()).throw(KeyError("missing")))
    response = client.post(
        "/api/chat", json={"thread_id": "missing", "message": "Explique"}
    )
    assert response.status_code == 404


def test_conversation_history_from_service(monkeypatch):
    monkeypatch.setattr(
        conversations, "load_conversation",
        lambda thread_id: {"thread_id": thread_id, "title": "Cours", "updated_at": None,
                           "messages": [{"role": "user", "content": "Question"}]},
    )
    response = client.get("/api/conversations/thread_1")
    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "Question"
