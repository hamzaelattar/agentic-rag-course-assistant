
import gc
import torch

from app.agent.llm import QwenLLM


def rewrite_query(
    question: str,
    history: list[dict] | None = None,
    previous_query: str = "",
    previous_answer: str = "",
    feedback: str = ""
) -> str:

    print("\n[REWRITER] Reformulation Qwen3...")

    history_text = "\n".join(
        f"{msg.get('role', 'user')}: "
        f"{msg.get('content', '')[:400]}"
        for msg in (history or [])[-4:]
    )

    prompt = f"""
Tu es un assistant spécialisé dans la
reformulation de questions pour un Agentic RAG.

Reformule la demande de l'étudiant afin de
faciliter sa compréhension par un Planner LLM.

Règles :
- Conserve toutes les intentions de l'étudiant.
- Ne supprime pas les demandes de recherche Web.
- Ne supprime pas les demandes de diagrammes.
- Conserve les numéros de page et noms propres.
- Résous les références à l'historique si possible.
- N'invente aucune information.
- Si la question est déjà claire, conserve-la.
- Pour une salutation, conserve la salutation.
- Retourne uniquement la question reformulée,
  sans commentaire.

Historique :
{history_text or "Aucun"}

Question originale :
{question}

Requête utilisée précédemment :
{previous_query or "Aucune"}

Réponse précédente :
{previous_answer[:900] or "Aucune"}

Problème constaté lors de la vérification :
{feedback or "Aucun"}

Question reformulée :
"""

    llm = QwenLLM()

    try:
        rewritten = llm.generate(prompt).strip()
    finally:
        del llm
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Conserver l'original si la sortie est vide
    if not rewritten:
        return question

    rewritten = rewritten.splitlines()[0].strip().strip('"')

    print("[REWRITER] Original :", question)
    print("[REWRITER] Nouvelle :", rewritten)

    return rewritten or question
