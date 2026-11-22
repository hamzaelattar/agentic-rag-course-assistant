
from app.rag.pipeline import RAGPipeline


def main():

    print("Démarrage du test RAG complet...")

    pipeline = RAGPipeline()

    question = (
        "Explique le mécanisme de self-attention "
        "et le rôle des matrices Q, K et V."
    )

    result = pipeline.answer(
        question,
        top_k=4
    )

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(result["question"])

    print("\n" + "=" * 60)
    print("RÉPONSE DE QWEN")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES UTILISÉES")
    print("=" * 60)

    for index, source in enumerate(
        result["sources"],
        start=1
    ):
        print(f"\nSource [{index}]")
        print("Document :", source["source"])
        print("Page :", source["page"])
        print("Chunk ID :", source["chunk_id"])
        print("Score :", source["rerank_score"])
        print("Recherche :", source["retrieved_via"])


if __name__ == "__main__":
    main()
