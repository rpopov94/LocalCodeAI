"""Парсер."""
from dataclasses import dataclass, field
from pathlib import Path

from langchain.document_loaders import DirectoryLoader, TextLoader, PythonLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


@dataclass
class RAGParser:
    project_path: Path
    documents: list = field(default_factory=list)
    splits: list = field(default_factory=list)
    vectorstore: object = None

    def load_project_files(self):
        """Load project files."""
        loaders = [
            DirectoryLoader(f"{self.project_path}/docs", glob="**/*.md", loader_cls=TextLoader),
            DirectoryLoader(f"{self.project_path}/src", glob="**/*.py", loader_cls=PythonLoader),
            TextLoader(f"{self.project_path}/README.md"),
        ]

        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        return documents


    def split_documents(self):
        """Split documents."""
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200,
        )

        markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        all_splits = []

        for doc in self.documents:
            if doc.metadata["source"].endswith(".py"):
                splits = python_splitter.split_documents([doc])
            elif doc.metadata["source"].endswith(".md"):
                splits = markdown_splitter.split_documents([doc])
            else:
                splits = text_splitter.split_documents([doc])

            all_splits.extend(splits)

        return all_splits

    def create_vector_store(self, persist_directory="vectorstore"):
        """Create vector store."""
        embeddings = OllamaEmbeddings(model="llama2")
        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        self.vectorstore.persist()


    def setup_rag(self):
        self.documents = self.load_project_files()
        self.split_documents()

        self.create_vector_store()

        llm = Ollama(model="llama2", temperature=0)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        return qa_chain

rag = RAGParser(project_path=Path('C:\WorkSpace\stw-timbrel'))

print(rag.setup_rag())