
import os
import numpy as np

from dotenv import load_dotenv
from neo4j import GraphDatabase


class Neo4jGraph:

    def __init__(self):
        load_dotenv()

        uri = os.getenv(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            raise ValueError("NEO4J_PASSWORD manquant")

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

        try:
            self.driver.verify_connectivity()

            self.driver.execute_query("""
                CREATE CONSTRAINT chunk_id_unique
                IF NOT EXISTS
                FOR (c:Chunk)
                REQUIRE c.id IS UNIQUE
            """)
        except Exception:
            self.driver.close()
            raise

        print("Connexion Neo4j réussie !")

    # 1. Création des noeuds Chunk
    def index_chunks(self, chunks):
        rows = [
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "chunk_index": chunk["metadata"]["chunk_index"]
            }
            for chunk in chunks
        ]

        self.driver.execute_query("""
            UNWIND $rows AS row

            MERGE (c:Chunk {id: row.id})

            SET c.text = row.text,
                c.source = row.source,
                c.page = row.page,
                c.chunk_index = row.chunk_index
        """, rows=rows)

        print(f"{len(rows)} chunks enregistrés")

    # 2. Relations de succession
    def link_adjacent(self, chunks):
        ordered = sorted(
            chunks,
            key=lambda c: (
                c["metadata"]["source"],
                c["metadata"]["page"],
                c["metadata"]["chunk_index"]
            )
        )

        edges = []

        for a, b in zip(ordered, ordered[1:]):
            if (
                a["metadata"]["source"]
                == b["metadata"]["source"]
            ):
                edges.append({
                    "from_id": a["id"],
                    "to_id": b["id"]
                })

        self.driver.execute_query("""
            UNWIND $edges AS edge

            MATCH (a:Chunk {id: edge.from_id})
            MATCH (b:Chunk {id: edge.to_id})

            MERGE (a)-[:NEXT]->(b)
        """, edges=edges)

        print(f"{len(edges)} liens NEXT traités")

    # 3. Relations sémantiques avec Qwen Embedding
    def link_similar(
        self,
        chunks,
        embeddings,
        top_k=2,
        threshold=0.65
    ):
        vectors = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError(
                "Dimensions embeddings/chunks invalides"
            )

        if not len(chunks):
            return

        # Normalisation cosine
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        vectors = vectors / np.maximum(norms, 1e-12)

        similarity_matrix = vectors @ vectors.T

        edges = {}

        for i, chunk in enumerate(chunks):
            ranked_indices = np.argsort(
                similarity_matrix[i]
            )[::-1]

            neighbors = 0

            for j in ranked_indices:
                if i == j:
                    continue

                other = chunks[j]

                if (
                    chunk["metadata"]["source"]
                    != other["metadata"]["source"]
                ):
                    continue

                # Exclure les voisins immédiats
                # d'une même page : déjà reliés par NEXT.
                same_page = (
                    chunk["metadata"]["page"]
                    == other["metadata"]["page"]
                )

                adjacent = (
                    abs(
                        chunk["metadata"]["chunk_index"]
                        - other["metadata"]["chunk_index"]
                    ) == 1
                )

                if same_page and adjacent:
                    continue

                score = float(similarity_matrix[i, j])

                if score < threshold:
                    continue

                # Paire canonique pour éviter
                # les relations sémantiques doublées
                a, b = sorted([
                    chunk["id"],
                    other["id"]
                ])

                edges[(a, b)] = max(
                    edges.get((a, b), -1),
                    score
                )

                neighbors += 1

                if neighbors >= top_k:
                    break

        rows = [
            {
                "from_id": a,
                "to_id": b,
                "score": score
            }
            for (a, b), score in edges.items()
        ]

        self.driver.execute_query("""
            UNWIND $edges AS edge

            MATCH (a:Chunk {id: edge.from_id})
            MATCH (b:Chunk {id: edge.to_id})

            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.score = edge.score
        """, edges=rows)

        print(f"{len(rows)} liens SIMILAR_TO traités")

    # 4. Recherche des chunks connectés
    def get_neighbors(self, chunk_id, limit=5):
        records, _, _ = self.driver.execute_query("""
            MATCH (c:Chunk {id: $chunk_id})
                  -[r:NEXT|SIMILAR_TO]-
                  (neighbor:Chunk)

            RETURN
                neighbor.id AS chunk_id,
                neighbor.text AS text,
                neighbor.page AS page,
                type(r) AS relation,
                r.score AS similarity

            LIMIT $limit
        """, chunk_id=chunk_id, limit=limit)

        return [record.data() for record in records]

    # 5. Statistiques du graphe
    def stats(self):
        queries = {
            "chunks": "MATCH (c:Chunk) RETURN count(c) AS n",
            "next": "MATCH ()-[r:NEXT]->() RETURN count(r) AS n",
            "similar": (
                "MATCH ()-[r:SIMILAR_TO]->() "
                "RETURN count(r) AS n"
            )
        }

        return {
            name: self.driver.execute_query(query)[0][0]["n"]
            for name, query in queries.items()
        }

    def close(self):
        self.driver.close()
