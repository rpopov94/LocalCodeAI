"""Base document parser."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PythonLoader, TextLoader
from langchain_core.documents import Document
from loguru import logger


class DocLoader:
    """Base document parser."""

    def __init__(self, source: str):
        self.source = Path(source).resolve()
        self._validate_path()

    def _validate_path(self):
        """Check file exists."""
        if not self.source.exists():
            raise FileNotFoundError(f"Path doesn't exist: {self.source}")

    def _load_files(self, glob_pattern: str, loader_cls, language: str) -> list[Document]:
        """Загрузка файлов определенного типа."""
        try:
            loader = DirectoryLoader(
                str(self.source),
                glob=glob_pattern,
                loader_cls=loader_cls,
                silent_errors=True,
                show_progress=True
            )
            docs = loader.load()
            logger.success(f"Loaded {len(docs)} {language} files")
            return docs
        except Exception as e:
            logger.error(f"Error load {language}: {e}")
            return []

    def parse_project(self) -> list[Document]:
        """Parallel load docs."""
        load_tasks = [
            ("**/*.[ch]pp", TextLoader, "C++"),
            ("**/*.py", PythonLoader, "Python"),
            ("**/*.js", TextLoader, "JavaScript")
        ]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda args: self._load_files(*args),
                load_tasks
            ))

        return [doc for sublist in results for doc in sublist]




