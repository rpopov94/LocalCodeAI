"""Local embeddings."""
from sentence_transformers import SentenceTransformer
import numpy as np


class LocalEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(f'./models/embedding/{model_name}')

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text)