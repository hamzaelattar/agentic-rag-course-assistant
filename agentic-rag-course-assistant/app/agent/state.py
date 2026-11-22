
from typing import TypedDict, Any


class AgentState(TypedDict, total=False):

    # Question originale de l'étudiant
    question: str

    # Reformulation avant le Planner
    rewritten_question: str

    # Historique (à connecter à SQLite plus tard)
    conversation_history: list[dict[str, str]]

    # Planification multi-outils
    selected_tools: list[str]
    remaining_tools: list[str]
    route: str
    tool_results: dict[str, Any]

    # Réponse
    answer: str
    sources: list[dict[str, Any]]
    diagram_code: str | None

    # Vérification et auto-correction
    answer_valid: bool | None
    quality_feedback: str
    rewrite_attempts: int
