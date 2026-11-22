
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.graph import build_agent


DATABASE_PATH = Path(
    "data/conversations.sqlite3"
)


@contextmanager
def open_conversation_agent():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with SqliteSaver.from_conn_string(
        str(DATABASE_PATH)
    ) as checkpointer:

        agent = build_agent(
            checkpointer=checkpointer
        )

        yield agent


def conversation_config(thread_id: str):

    if not thread_id.strip():
        raise ValueError(
            "thread_id ne peut pas être vide"
        )

    return {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 40
    }


def ask_agent(
    agent,
    thread_id: str,
    question: str
):

    if not question.strip():
        raise ValueError(
            "La question ne peut pas être vide"
        )

    # Réinitialiser les données propres au tour.
    # Ne pas réinitialiser conversation_history :
    # LangGraph le restaure depuis SQLite.
    initial_state = {
        "question": question.strip(),
        "rewritten_question": "",
        "selected_tools": [],
        "remaining_tools": [],
        "route": "",
        "tool_results": {},
        "answer": "",
        "sources": [],
        "diagram_code": None,
        "answer_valid": None,
        "quality_feedback": "",
        "rewrite_attempts": 0
    }

    return agent.invoke(
        initial_state,
        config=conversation_config(thread_id)
    )


def get_conversation_history(
    agent,
    thread_id: str
):

    snapshot = agent.get_state(
        conversation_config(thread_id)
    )

    if not snapshot.values:
        return []

    return snapshot.values.get(
        "conversation_history", []
    )
