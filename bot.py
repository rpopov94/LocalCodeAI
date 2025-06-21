from typing import Dict, List

import ollama

from db import collection


class RAGBot:
    def __init__(self, model_name: str = "deepseek-llm:7b"):
        self.model = model_name
        self.chat_history: List[Dict[str, str]] = []

    def retrieve_context(self, query: str, n_results: int = 2) -> str:
        results = collection.query(query_texts=[query], n_results=n_results)
        return "\n".join(results["documents"][0])

    def generate_response(self, query: str) -> str:
        context = self.retrieve_context(query)

        prompt = f"""
        Контекст: {context}

        История диалога:
        {self._format_history()}

        Вопрос: {query}
        Ответ:"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7}
        )

        self._update_history(query, response["message"]["content"])
        return response["message"]["content"]

    def _format_history(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history])

    def _update_history(self, user_query: str, bot_response: str):
        self.chat_history.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": bot_response}
        ])
        self.chat_history = self.chat_history[-10:]
