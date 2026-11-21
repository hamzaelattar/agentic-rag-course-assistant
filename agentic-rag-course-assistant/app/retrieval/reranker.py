
import gc
import torch
from sentence_transformers import CrossEncoder


class QwenReranker:

    def __init__(self):
        self.model_name = "Qwen/Qwen3-Reranker-0.6B"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponible")

        print("Chargement Qwen3 Reranker sur CUDA...")

        self.model = CrossEncoder(
            self.model_name,
            device="cuda",
            max_length=512,
            model_kwargs={
                "torch_dtype": torch.float16
            }
        )

        print("Qwen3 Reranker chargé sur GPU !")

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []

        pairs = [
            (query, doc["text"])
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=1,
            show_progress_bar=True
        )

        ranked = [
            {
                **doc,
                "rerank_score": float(score)
            }
            for doc, score in zip(documents, scores)
        ]

        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked[:top_k]

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        print("Reranker libéré du GPU")
