"""Local llm."""
# core/llm/local_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import Optional


class LocalLLM:
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация локальной LLM.

        Args:
            model_path: Путь к локальной модели или идентификатор модели из HuggingFace Hub
                       Если None, будет использована модель по умолчанию
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path is None:
            model_path = "microsoft/phi-2"
            self.local = False
        else:
            self.local = True

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=self.local
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                local_files_only=self.local
            ).to(self.device)

        except Exception as e:
            raise RuntimeError(f"Can't loading model: {e}")

    def generate(self, prompt: str, max_length: int = 500) -> str:
        """Generate by prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Generation error: {e}")
