
import gc
import torch
from sentence_transformers import SentenceTransformer


class QwenEmbeddings:

    def __init__(self):
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.device = "cuda"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponible")

        print("Chargement Qwen3 Embedding sur CUDA...")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={
                "torch_dtype": torch.float16
            }
        )

        self.model.max_seq_length = 512

        print("Qwen3 Embedding chargé sur GPU !")

    def embed_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        print("Embedding libéré du GPU")
