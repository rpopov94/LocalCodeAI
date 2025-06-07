"""Chroma manager."""
import chromadb
from typing import List, Dict
import numpy as np
import os


class ChromaDBManager:
    def __init__(self, collection_name: str = "code_embeddings", persist_dir: str = "./chroma_db"):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: List[str], embeddings: List[np.ndarray],
               metadatas: List[Dict], documents: List[str]):
        """Добавляет или обновляет векторы в коллекции"""
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Query by vector space."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return formatted_results

    def clear_collection(self):
        """Full clear collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(self.collection.name)
