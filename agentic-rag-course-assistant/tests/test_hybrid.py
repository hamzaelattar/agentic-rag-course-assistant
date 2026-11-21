
import torch

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunking import create_chunks
from app.retrieval.embeddings import QwenEmbeddings
from app.retrieval.qdrant_store import HybridQdrant
from app.retrieval.reranker import QwenReranker


def gpu_memory():
    used = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM PyTorch utilisée : {used:.2f} Go")


# 1. Charger le PDF
pages = load_pdf("data/cours_transformers.pdf")
chunks = create_chunks(pages)

print("Chunks :", len(chunks))

question = "Explique le mécanisme de self-attention"

# 2. Embeddings sur CUDA
embedder = QwenEmbeddings()

texts = [chunk["text"] for chunk in chunks]
vectors = embedder.embed_documents(texts)

query_vector = embedder.embed_query(question)

print("Dimensions :", vectors.shape)
gpu_memory()

# 3. Qdrant Hybrid Search
store = HybridQdrant()
store.index_chunks(chunks, vectors)

results = store.search(
    question,
    query_vector,
    limit=8
)

print("Résultats hybrides :", len(results))
store.client.close()

# 4. Libérer Embedding
embedder.unload()
del embedder

gpu_memory()

# 5. Reranking sur CUDA
reranker = QwenReranker()

final_results = reranker.rerank(
    question,
    results,
    top_k=5
)

gpu_memory()

# 6. Afficher les résultats
for index, doc in enumerate(final_results, 1):
    print("\n" + "=" * 50)
    print("Rang :", index)
    print("Page :", doc["page"])
    print("Score :", doc["rerank_score"])
    print("Texte :", doc["text"][:400])

# 7. Libérer Reranker
reranker.unload()
del reranker

gpu_memory()
