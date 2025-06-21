from chromadb import Documents, EmbeddingFunction, HttpClient
from typing import List


class MiniLMEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> List[List[float]]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(texts).tolist()

client = HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=MiniLMEmbeddingFunction()
)

documents = [
    "DeepSeek-R1 поддерживает 128K контекста.",
    "DeepSeek-Coder умеет анализировать Python, C++ и JavaScript.",
    "Ollama — это инструмент для запуска LLM локально."
]
collection.add(documents=documents, ids=[f"doc_{i}" for i in range(len(documents))])
