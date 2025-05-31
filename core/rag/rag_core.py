"""Core of rag."""
from typing import List, Dict

from langchain_community.docstore import document

from core.embeddings.local_ebeddings import LocalEmbedder
from core.parser.python import PythonCodeParser
from core.vector_db.chroma_manager import ChromaDBManager


class RAGSystem:
    def __init__(self, repo_path: str):
        self.parser = PythonCodeParser(repo_path)
        self.embedder = LocalEmbedder()
        self.vector_db = ChromaDBManager()

    def build_knowledge_base(self):
        """Полный цикл создания базы знаний"""
        entities = self.parser.parse_project()

        documents = []
        metadatas = []
        ids = []

        for i, entity in enumerate(entities):
            content = f"{entity['type']} {entity['name']}:\n{document['docstring']}\nCode:\n{entity['code']}"
            documents.append(content)
            metadatas.append({
                'type': entity['type'],
                'file': entity['file']
            })
            ids.append(str(i))

        embeddings = self.embedder.embed_batch(documents)
        self.vector_db.upsert(ids, embeddings, metadatas, documents)

    def query(self, question: str, top_k: int = 3) -> List[Dict]:
        """Поиск по базе знаний"""
        query_embedding = self.embedder.embed(question)
        return self.vector_db.search(query_embedding, top_k)