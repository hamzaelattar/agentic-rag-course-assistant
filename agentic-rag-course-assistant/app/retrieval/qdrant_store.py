
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


class HybridQdrant:

    def __init__(self):
        self.collection = "course_hybrid"
        self.client = QdrantClient(path="data/qdrant")

        self.bm25 = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )

    @staticmethod
    def to_sparse(vector):
        return models.SparseVector(
            indices=vector.indices.tolist(),
            values=vector.values.tolist()
        )

    def index_chunks(self, chunks, dense_vectors):
        if len(chunks) != len(dense_vectors):
            raise ValueError("Nombre de chunks/vecteurs différent")

        texts = [chunk["text"] for chunk in chunks]
        sparse_vectors = list(self.bm25.embed(texts))

        points = []

        for chunk, dense, sparse in zip(
            chunks, dense_vectors, sparse_vectors
        ):
            # Qdrant accepte les UUID, mais pas nos ID
            # SHA-256 tronqués sous forme de chaînes.
            point_id = str(
                uuid5(NAMESPACE_URL, chunk["id"])
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense.tolist(),
                        "sparse": self.to_sparse(sparse)
                    },
                    payload={
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True
        )

        print(f"{len(points)} chunks indexés dans Qdrant")

    def search(self, query, dense_vector, limit=8):
        sparse_vector = next(
            iter(self.bm25.query_embed(query))
        )

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector.tolist(),
                    using="dense",
                    limit=limit
                ),
                models.Prefetch(
                    query=self.to_sparse(sparse_vector),
                    using="sparse",
                    limit=limit
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit,
            with_payload=True
        )

        return [
            {
                **point.payload,
                "score_hybrid": point.score
            }
            for point in results.points
        ]
