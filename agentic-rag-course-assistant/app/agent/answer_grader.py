
import gc
import torch

from app.agent.llm import QwenLLM


def grade_answer(
    question: str,
    answer: str,
    sources: list[dict],
    selected_tools: list[str],
    evidence: str = "",
    diagram_code: str | None = None
) -> dict:

    print("\n[GRADER] Vérification de la réponse...")

    if not answer.strip():
        return {
            "answer_valid": False,
            "quality_feedback": "La réponse est vide."
        }

    documentary_tools = {
        "course_rag",
        "document_lookup",
        "web_search"
    }

    if documentary_tools.intersection(selected_tools) and not sources:
        return {
            "answer_valid": False,
            "quality_feedback": (
                "Aucune source documentaire disponible."
            )
        }

    if "diagram_generator" in selected_tools:
        if not diagram_code or not diagram_code.startswith(
            ("flowchart ", "graph ")
        ):
            return {
                "answer_valid": False,
                "quality_feedback": (
                    "Diagramme Mermaid manquant ou mal formé."
                )
            }

    prompt = f"""
Tu es un évaluateur de réponses pédagogiques.

Décide si la réponse satisfait la demande
de l'étudiant.

Critères :
1. Réponse pertinente pour la question.
2. Pas de contradiction manifeste avec
   les éléments justificatifs fournis.
3. Les parties importantes de la demande
   sont traitées.
4. Si une recherche documentaire est demandée,
   la réponse doit être appuyée sur des sources.
5. Si un diagramme est demandé, celui-ci doit
   correspondre au sujet.

Réponds uniquement :
A = réponse suffisamment pertinente
B = réponse insuffisante ou incorrecte

QUESTION :
{question[:500]}

RÉPONSE :
{answer[:1200]}

ÉLÉMENTS JUSTIFICATIFS :
{evidence[:1600] or "Aucun"}

OUTILS :
{", ".join(selected_tools)}

CATÉGORIE :
"""

    llm = QwenLLM()

    try:
        tokenizer = llm.tokenizer
        model = llm.model

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu évalues la pertinence des réponses. "
                    "Réponds uniquement par A ou B."
                )
            },
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        a_ids = tokenizer.encode(
            "A", add_special_tokens=False
        )
        b_ids = tokenizer.encode(
            "B", add_special_tokens=False
        )

        if len(a_ids) != 1 or len(b_ids) != 1:
            raise ValueError("Labels A/B incompatibles")

        with torch.inference_mode():
            logits = model(
                **inputs,
                use_cache=False
            ).logits[0, -1]

            scores = torch.stack([
                logits[a_ids[0]],
                logits[b_ids[0]]
            ]).float()

            probability = torch.softmax(
                scores, dim=0
            )[0].item()

        valid = probability >= 0.5

        print(
            f"[GRADER] Pertinence estimée : "
            f"{probability:.2%}"
        )

        return {
            "answer_valid": valid,
            "quality_feedback": (
                "Réponse jugée suffisamment pertinente."
                if valid else
                "La réponse ne satisfait pas suffisamment "
                "la demande. Reformuler la recherche."
            )
        }

    finally:
        del llm
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
