
import gc
import re
from difflib import SequenceMatcher
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

    # Qwen peut recopier le libellé final du prompt au lieu de
    # reformuler la question. Dans ce cas, garder la demande originale.
    rewritten = re.sub(
        r"^(?:la\s+)?(?:question|requête)\s+(?:reformulée|utilisée)\s*:\s*",
        "",
        rewritten,
        flags=re.IGNORECASE,
    ).strip()

    if (
        not rewritten
        or len(rewritten) < 4
        or rewritten.casefold().strip(" .:!") in {"aucun", "aucune", "none", "n/a"}
    ):
        rewritten = question

    # Ne pas reprendre une ancienne rÃ©ponse comme nouvelle question.
    normalized_rewrite = re.sub(r"\W+", " ", rewritten.casefold()).strip()
    for message in (history or [])[-6:]:
        if message.get("role") != "assistant":
            continue

        old_answer = re.sub(
            r"\W+", " ", message.get("content", "").casefold()
        ).strip()
        if not old_answer:
            continue

        old_prefix = old_answer[:max(len(normalized_rewrite) * 2, 160)]
        if (
            normalized_rewrite in old_answer
            or SequenceMatcher(None, normalized_rewrite, old_prefix).ratio() >= 0.72
        ):
            rewritten = question
            break

    print("[REWRITER] Original :", question)
    print("[REWRITER] Nouvelle :", rewritten)

    return rewritten or question
