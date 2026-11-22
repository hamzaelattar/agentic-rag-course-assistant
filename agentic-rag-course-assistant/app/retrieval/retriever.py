
from app.retrieval.embeddings import QwenEmbeddings
from app.retrieval.qdrant_store import HybridQdrant
from app.retrieval.neo4j_store import Neo4jGraph
from app.retrieval.graph_search import GraphSearch


class HybridGraphRetriever:

    def retrieve(
        self,
        question: str,
        hybrid_limit: int = 8,
        max_graph_chunks: int = 8
    ) -> list[dict]:

        print("\n[1] Qwen3 Embedding — CUDA")

        embedder = QwenEmbeddings()

        try:
            query_vector = embedder.embed_query(question)
        finally:
            embedder.unload()

        print("[2] Qdrant — Dense + BM25 + RRF")

        store = HybridQdrant()

        try:
            hybrid_results = store.search(
                question,
                query_vector,
                limit=hybrid_limit
            )
        finally:
            store.client.close()

        print(
            "Résultats Qdrant :",
            len(hybrid_results)
        )

        if not hybrid_results:
            raise RuntimeError(
                "Aucun résultat dans Qdrant. "
                "Exécute d'abord tests.test_hybrid "
                "pour indexer le cours."
            )

        print("[3] Neo4j — Graph Expansion")

        graph = Neo4jGraph()

        try:
            graph_search = GraphSearch(graph)

            documents = graph_search.expand(
                hybrid_results,
                neighbors_per_seed=2,
                max_graph_chunks=max_graph_chunks
            )
        finally:
            graph.close()

        print(
            "Total de chunks après expansion :",
            len(documents)
        )

        return documents
