
def main():
    print("Démarrage du test GraphRAG...", flush=True)

    from app.ingestion.pdf_loader import load_pdf
    from app.ingestion.chunking import create_chunks
    from app.retrieval.embeddings import QwenEmbeddings
    from app.retrieval.neo4j_store import Neo4jGraph

    # 1. Charger le cours
    pages = load_pdf("data/cours_transformers.pdf")
    chunks = create_chunks(pages)

    print("Nombre de chunks :", len(chunks))

    # 2. Embeddings CUDA
    embedder = QwenEmbeddings()

    try:
        vectors = embedder.embed_documents(
            [chunk["text"] for chunk in chunks]
        )
    finally:
        embedder.unload()

    print("Dimensions embeddings :", vectors.shape)

    # 3. Construire le graphe
    graph = Neo4jGraph()

    try:
        graph.index_chunks(chunks)
        graph.link_adjacent(chunks)

        graph.link_similar(
            chunks,
            vectors,
            top_k=2,
            threshold=0.65
        )

        # 4. Statistiques
        stats = graph.stats()

        print("\n===== GRAPH STATS =====")
        print("Chunks :", stats["chunks"])
        print("NEXT :", stats["next"])
        print("SIMILAR_TO :", stats["similar"])

        # 5. Tester la navigation
        chunk_id = chunks[0]["id"]

        neighbors = graph.get_neighbors(
            chunk_id,
            limit=5
        )

        print("\n===== GRAPH SEARCH =====")

        for neighbor in neighbors:
            print("\nChunk :", neighbor["chunk_id"])
            print("Page :", neighbor["page"])
            print("Relation :", neighbor["relation"])
            print("Texte :", neighbor["text"][:200])

    finally:
        graph.close()


if __name__ == "__main__":
    main()
