
import sys


def main():
    print("Démarrage Agentic RAG LangGraph...", flush=True)

    from app.agent.graph import build_agent

    agent = build_agent()

    # Récupérer la question
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = (
            "Consulte la page 3 du cours, "
            "recherche des informations sur Internet "
            "et génère un diagramme explicatif."
        )

    print("\n" + "=" * 60)
    print("QUESTION ORIGINALE")
    print("=" * 60)
    print(question)

    # Exécuter l'agent LangGraph
    result = agent.invoke(
        {
            "question": question,
            "rewrite_attempts": 0,
            "conversation_history": []
        },
        config={
            "recursion_limit": 40
        }
    )

    # ======================================
    # QUERY REWRITING
    # ======================================

    print("\n" + "=" * 60)
    print("QUERY REWRITING")
    print("=" * 60)

    print(
        "Question reformulée :",
        result.get("rewritten_question", question)
    )

    # ======================================
    # PLANNER MULTI-OUTILS
    # ======================================

    print("\n" + "=" * 60)
    print("PLANIFICATION DES OUTILS")
    print("=" * 60)

    print("Plan :", result.get("selected_tools", []))
    print("Route :", result.get("route", "Non définie"))

    # ======================================
    # REPONSE FINALE
    # ======================================

    print("\n" + "=" * 60)
    print("RÉPONSE DE L'AGENT")
    print("=" * 60)

    print(result.get("answer", ""))

    # ======================================
    # SOURCES
    # ======================================

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    sources = result.get("sources", [])

    if not sources:
        print("Aucune source utilisée.")

    for index, source in enumerate(sources, start=1):

        print(f"\nSource [{index}]")

        if source.get("source"):
            print("PDF :", source["source"])

        if source.get("page") is not None:
            print("Page :", source["page"])

        if source.get("chunk_id"):
            print("Chunk ID :", source["chunk_id"])

        if source.get("title"):
            print("Titre :", source["title"])

        if source.get("url"):
            print("URL :", source["url"])

        if source.get("retrieved_via"):
            print(
                "Recherche :",
                source["retrieved_via"]
            )

    # ======================================
    # DIAGRAM GENERATOR
    # ======================================

    if result.get("diagram_code"):

        print("\n" + "=" * 60)
        print("DIAGRAMME MERMAID")
        print("=" * 60)

        print(result["diagram_code"])

    # ======================================
    # ANSWER GRADING
    # ======================================

    print("\n" + "=" * 60)
    print("ÉVALUATION DE LA RÉPONSE")
    print("=" * 60)

    print(
        "Réponse valide :",
        result.get("answer_valid")
    )

    print(
        "Feedback :",
        result.get("quality_feedback", "")
    )

    # ======================================
    # SELF-CORRECTION LOOP
    # ======================================

    print("\n" + "=" * 60)
    print("SELF-CORRECTION")
    print("=" * 60)

    print(
        "Nombre de reformulations supplémentaires :",
        result.get("rewrite_attempts", 0)
    )

    print("\nTest terminé !")


if __name__ == "__main__":
    main()
