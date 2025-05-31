"""Local embeddings."""
from sentence_transformers import SentenceTransformer
from torch import Tensor


class LocalEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(f'./models/embedding/{model_name}')

    def embed(self, text: str) -> Tensor:
        return self.model.encode(text)

    def embed_batch(self, texts: list[str]):
        """Реализация для батча текстов"""
        return [self.embed(text) for text in texts]
