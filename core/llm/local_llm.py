"""Local LLM wrapper with improved error handling and generation parameters."""
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LocalLLM:
    def __init__(self, model_path: str = "microsoft/phi-2", device: str = "auto"):
        """
        Initialize local LLM with configurable model path and device.

        Args:
            model_path: Path or name of pretrained model
            device: Target device ('cpu', 'cuda' or 'auto')
        """
        transformers.logging.set_verbosity_error()
        self.device = device if torch.cuda.is_available() and device == "auto" else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"  # For batch processing
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 200,
            temperature: float = 0.7,
            top_p: float = 0.9,
            repetition_penalty: float = 1.1,
            do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt with configurable parameters.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Creativity control (0-1)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            do_sample: Enable sampling

        Returns:
            Generated text
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            ).to(self.device)

            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]
