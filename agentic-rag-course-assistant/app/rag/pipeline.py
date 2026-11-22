
import gc
import torch

from app.agent.llm import QwenLLM
from app.retrieval.retriever import HybridGraphRetriever
from app.retrieval.reranker import QwenReranker


class RAGPipeline:

    def __init__(self):
        self.retriever = HybridGraphRetriever()

    def answer(self, question: str, top_k: int = 4):
        if not question.strip():
            raise ValueError("La question ne peut pas être vide")

        print("\n===== RAG PIPELINE =====")

        # 1. Recherche hybride + expansion graphe
        documents = self.retriever.retrieve(
            question,
            hybrid_limit=8,
            max_graph_chunks=8
        )

        # 2. Reranking sur GPU
        print("\n[4] Qwen3 Reranker — CUDA")

        reranker = QwenReranker()

        try:
            ranked_documents = reranker.rerank(
                question,
                documents,
                top_k=top_k
            )
        finally:
            reranker.unload()

        # 3. Construire le contexte avec les sources
        context_parts = []

        for index, doc in enumerate(
            ranked_documents,
            start=1
        ):
            context_parts.append(
                f"[{index}] "
                f"Document : {doc['source']}\n"
                f"Page : {doc['page']}\n"
                f"Contenu :\n{doc['text'][:900]}"
            )

        context = "\n\n".join(context_parts)

        # 4. Préparer le prompt RAG
        prompt = f"""
Tu es un assistant pédagogique spécialisé
dans les Transformers et les LLMs.

Réponds en français à la question de l'étudiant.

Règles :
- Utilise uniquement les extraits du cours fournis.
- Explique clairement et simplement.
- N'invente pas d'informations absentes du contexte.
- Cite les passages utilisés avec [1], [2], etc.
- Si le contexte est insuffisant, indique-le.

CONTEXTE DU COURS :
{context}

QUESTION :
{question}

RÉPONSE :
"""

        # 5. Génération avec Qwen3 sur CUDA
        print("\n[5] Qwen3 LLM — CUDA")

        llm = QwenLLM()

        try:
            response = llm.generate(prompt)
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()

        # 6. Retourner la réponse et ses sources
        sources = [
            {
                "chunk_id": doc["chunk_id"],
                "source": doc["source"],
                "page": doc["page"],
                "rerank_score": doc["rerank_score"],
                "retrieved_via": doc["retrieved_via"],
                "text": doc["text"]
            }
            for doc in ranked_documents
        ]

        return {
            "question": question,
            "answer": response,
            "sources": sources
        }
