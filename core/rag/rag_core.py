"""Core of rag."""
from typing import List, Dict

from core.embeddings.local_ebeddings import LocalEmbedder
from core.parser.docs import DocLoader
from core.vector_db.chroma_manager import ChromaDBManager


class RAGSystem:
    """Simple rag system."""

    def __init__(self, repo_path: str):
        self.parser = DocLoader(repo_path)
        self.embedder = LocalEmbedder()
        self.vector_db = ChromaDBManager()

    def build_knowledge_base(self):
        """The full cycle of knowledge base creation."""
        entities = self.parser.parse_project()

        documents = []
        metalist = []
        ids = []

        for i, entity in enumerate(entities):
            content = f"{entity.type} \nCode:\n{entity.page_content}"
            documents.append(content)
            metalist.append({
                'type': entity.type,
                'file': entity.metadata.get('source', None),
            })
            ids.append(str(i))

        embeddings = self.embedder.embed_batch(documents)
        self.vector_db.upsert(ids, embeddings, metalist, documents)

    def query(self, question: str, top_k: int = 3) -> List[Dict]:
        """Knowledge base search."""
        query_embedding = self.embedder.embed(question)
        return self.vector_db.search(query_embedding, top_k)
