
class GraphSearch:
    def __init__(self, graph):
        self.graph = graph

    def expand(
        self,
        seed_chunks: list[dict],
        neighbors_per_seed: int = 2,
        max_graph_chunks: int = 8
    ) -> list[dict]:

        # Résultats initiaux de Qdrant
        documents = {
            doc["chunk_id"]: {
                **doc,
                "retrieved_via": ["hybrid"]
            }
            for doc in seed_chunks
        }

        added = 0

        for seed in seed_chunks:
            if added >= max_graph_chunks:
                break

            neighbors = self.graph.get_neighbors(
                chunk_id=seed["chunk_id"],
                limit=10
            )

            # Favoriser NEXT, puis les liens SIMILAR_TO
            neighbors = sorted(
                neighbors,
                key=lambda n: (
                    n["relation"] != "NEXT",
                    -(n.get("similarity") or 0.0)
                )
            )

            for neighbor in neighbors[:neighbors_per_seed]:
                chunk_id = neighbor["chunk_id"]

                if chunk_id in documents:
                    methods = documents[chunk_id][
                        "retrieved_via"
                    ]

                    if "graph" not in methods:
                        methods.append("graph")

                    continue

                if added >= max_graph_chunks:
                    break

                documents[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": neighbor["text"],
                    "source": seed["source"],
                    "page": neighbor["page"],
                    "score_hybrid": None,
                    "graph_relation": neighbor["relation"],
                    "graph_similarity": neighbor.get(
                        "similarity"
                    ),
                    "graph_seed_id": seed["chunk_id"],
                    "retrieved_via": ["graph"]
                }

                added += 1

        return list(documents.values())
