"""Local embeddings."""
from pathlib import Path

from sentence_transformers import SentenceTransformer
from torch import Tensor


class LocalEmbedder:
    """Local llm embedder."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        model_path = Path('.') / 'models' / 'embeddings' / model_name
        self.model = SentenceTransformer(str(model_path))

    def embed(self, text: str) -> Tensor:
        """Embeded text."""
        return self.model.encode(text)

    def embed_batch(self, texts: list[str]):
        """Text embedded batches."""
        return [self.embed(text) for text in texts]
