
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLLM:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-0.6B"
        self.device = "cuda"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA n'est pas disponible")

        print("Chargement de Qwen3-0.6B...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16
        ).to(self.device)

        self.model.eval()

        print("Modèle chargé sur GPU !")

    def generate(self, question):
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant pédagogique spécialisé "
                    "dans les Transformers et les LLMs. "
                    "Réponds clairement en français."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_tokens = outputs[0][
            inputs["input_ids"].shape[-1]:
        ]

        return self.tokenizer.decode(
            response_tokens,
            skip_special_tokens=True
        ).strip()
