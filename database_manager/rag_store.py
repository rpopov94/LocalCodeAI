import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.vectorstores import VectorStore
from loguru import logger

from database_manager.types import Embedding


class Mananer:
    """Класс для обработки мультиязычных проектов (C++ и Python) и создания RAG системы."""

    def __init__(self, project_path: str, embeddings: Embedding):
        """
        Инициализация парсера.

        Args:
            project_path: Путь к проекту (строка или Path-like)
        """
        self.project_path = Path(project_path).resolve()
        self.splits: List[Document] = []
        self.vectorstore: Optional[VectorStore] = None

        self._embeddings = embeddings

        # Настройка предупреждений
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

    def load_documents(self) -> List[Document]:
        """Загрузка документов проекта с обработкой C++ и Python файлов."""
        try:
            if not self.project_path.exists():
                raise FileNotFoundError(f"Путь не существует: {self.project_path}")

            logger.debug(f"Ищем файлы по пути: {self.project_path}")

            cpp_loader = DirectoryLoader(
                str(self.project_path),
                glob="**/*.[ch]pp",
                loader_cls=TextLoader,
                silent_errors=True,
                show_progress=True
            )
            cpp_docs = cpp_loader.load()
            logger.info(f"Загружено {len(cpp_docs)} C++ файлов")

            py_loader = DirectoryLoader(
                str(self.project_path),
                glob="**/*.py",
                loader_cls=PythonLoader,
                silent_errors=True,
                show_progress=True
            )
            py_docs = py_loader.load()
            logger.info(f"Загружено {len(py_docs)} Python файлов")

            js_loader = DirectoryLoader(
                str(self.project_path),
                glob="**/*.js",
                loader_cls=TextLoader,
                silent_errors=True,
                show_progress=True
            )

            js_docs = js_loader.load()
            logger.info(f"Загружено {len(js_docs)} JS файлов")

            return cpp_docs + py_docs + js_docs

        except Exception as e:
            print(f"[ERROR] Ошибка загрузки документов: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Разбиение документов на чанки с учетом языка."""
        if not documents:
            print("[WARNING] Нет документов для разбиения")
            return []

        try:
            cpp_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.CPP,
                chunk_size=1000,
                chunk_overlap=200
            )
            py_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=1000,
                chunk_overlap=200
            )
            js_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.JS,
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = []
            for doc in documents:
                if doc.metadata['source'].endswith(('.cpp', '.hpp', '.h', '.cxx')):
                    splits.extend(cpp_splitter.split_documents([doc]))
                elif doc.metadata['source'].endswith('.py'):
                    splits.extend(py_splitter.split_documents([doc]))
                elif doc.metadata['source'].endwith('.js'):
                    splits.extend(js_splitter.split_documents([doc]))
                else:
                    general_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150
                    )
                    splits.extend(general_splitter.split_documents([doc]))

            print(f"[INFO] Документы разбиты на {len(splits)} чанков")
            return splits

        except Exception as e:
            print(f"[ERROR] Ошибка разбиения документов: {e}")
            return []

    def create_vector_store(self, persist_directory: str = "vectorstore") -> bool:
        """Создание векторного хранилища."""
        if not self.splits:
            print("[ERROR] Нет данных для создания векторного хранилища")
            return False

        try:
            self.vectorstore = Chroma.from_documents(
                documents=self.splits,
                embedding=self.embeddings,
                persist_directory=str(persist_directory),
                collection_metadata={"hnsw:space": "cosine"}
            )

            print(f"[INFO] Векторное хранилище создано в {persist_directory}")
            return True

        except Exception as e:
            print(f"[ERROR] Ошибка создания векторного хранилища: {e}")
            return False

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Выполнение запроса к RAG системе."""
        if not self.vectorstore:
            return {"error": "Векторное хранилище не инициализировано"}

        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(question)

            return {
                "question": question,
                "sources": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0)
                } for doc in docs]
            }
        except Exception as e:
            return {"error": str(e)}

    def setup_rag(self) -> Dict[str, Any]:
        """Полная настройка RAG системы."""
        print("[INFO] Начало настройки RAG системы...")

        documents = self.load_documents()
        if not documents:
            return {"status": "error", "message": "Не удалось загрузить документы"}

        self.splits = self.split_documents(documents)
        if not self.splits:
            return {"status": "error", "message": "Не удалось разбить документы"}

        if not self.create_vector_store():
            return {"status": "error", "message": "Не удалось создать векторное хранилище"}

        return {"status": "success", "message": "RAG система успешно настроена"}


def main():
    """Точка входа."""
    parser = Mananer(
        project_path=r"C:\WorkSpace\orbit-viewer",
        embeddings=OllamaEmbeddings(model="all-minilm"),
    )

    result = parser.setup_rag()
    print("\nРезультат настройки:", result)

    if result["status"] == "success":
        # Запрос по C++ коду
        cpp_query = parser.query("Как работает класс DataProcessor в C++?")
        print("\nРезультат запроса по C++:")
        for idx, source in enumerate(cpp_query["sources"], 1):
            print(f"\n--- Источник {idx} ---")
            print(f"Файл: {source['metadata']['source']}")
            print(f"Содержимое:\n{source['content'][:200]}...")

        py_query = parser.query("Как используется функция process_data в Python?")
        print("\nРезультат запроса по Python:")
        for idx, source in enumerate(py_query["sources"], 1):
            print(f"\n--- Источник {idx} ---")
            print(f"Файл: {source['metadata']['source']}")
            print(f"Содержимое:\n{source['content'][:200]}...")


if __name__ == "__main__":
   main()
