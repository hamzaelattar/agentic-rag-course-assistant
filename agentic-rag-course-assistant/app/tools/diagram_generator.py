import gc
import re

import torch

from app.agent.llm import QwenLLM


def clean_mermaid(response: str) -> str:
    """Extraire le code Mermaid généré par Qwen."""

    response = response.strip()

    # Retirer les blocs Markdown si présents
    match = re.search(
        r"```(?:mermaid)?\s*(.*?)```",
        response,
        flags=re.DOTALL | re.IGNORECASE
    )

    if match:
        response = match.group(1).strip()

    # Vérifier le début du diagramme
    valid_start = re.match(
        r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b",
        response,
        flags=re.IGNORECASE
    )

    if not valid_start:
        raise ValueError(
            "Qwen n'a pas retourné un diagramme Mermaid "
            "au format flowchart attendu."
        )

    return response


def generate_diagram(question: str) -> dict:
    """Générer un diagramme Mermaid avec Qwen3."""

    print("\n[TOOL] Diagram Generator — Qwen3 CUDA")

    if not question.strip():
        raise ValueError("La demande est vide")

    llm = QwenLLM()

    try:
        prompt = f"""
Tu es un expert en création de diagrammes Mermaid.

Ta mission est de transformer la demande
de l'étudiant en diagramme pédagogique.

RÈGLES OBLIGATOIRES :

1. Génère uniquement du code Mermaid.
2. Commence toujours par : flowchart TD
3. Utilise entre 4 et 8 noeuds.
4. Utilise des identifiants simples : A, B, C...
5. Les relations doivent être cohérentes.
6. Utilise des labels courts en français.
7. N'écris aucune explication.
8. N'utilise pas de balises Markdown.

FORMAT ATTENDU :

flowchart TD
    A["Question"] --> B["Recherche"]
    B --> C["Analyse"]
    C --> D["Réponse"]

DEMANDE DE L'ÉTUDIANT :
{question}

CODE MERMAID :
"""

        # Deux tentatives si le format est incorrect
        last_error = None

        for attempt in range(2):
            response = llm.generate(prompt)

            try:
                diagram_code = clean_mermaid(response)

                print("[DIAGRAM] Code Mermaid généré")

                return {
                    "diagram_code": diagram_code,
                    "format": "mermaid"
                }

            except ValueError as exc:
                last_error = exc
                print(
                    "[DIAGRAM] Format incorrect, tentative",
                    attempt + 1
                )

                prompt += (
                    "\nRappel : retourne uniquement "
                    "un diagramme commençant par "
                    "'flowchart TD'.\n"
                )

        raise ValueError(
            f"Échec de génération Mermaid : {last_error}"
        )

    finally:
        del llm
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
