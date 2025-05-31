"""Local llm."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LocalLLM:
    def __init__(self, model_path: str = "microsoft/phi-2"):
        self.device = "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to('cpu')

        except Exception as e:
            raise RuntimeError(f"Can't load model: {str(e)}")

    def generate(self, prompt: str, max_length: int = 200) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cpu')

            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            raise RuntimeError(f"Generation error: {str(e)}")