import os

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PythonLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

doc_path = os.getenv("DATA_DIR")

loaders = [
    # DirectoryLoader(doc_path, glob="**/*.txt", loader_cls=TextLoader),
    DirectoryLoader(doc_path, glob="**/*.py", loader_cls=PythonLoader),
    # DirectoryLoader(doc_path, glob="**/*.rst", loader_cls=UnstructuredFileLoader),
    DirectoryLoader(doc_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

vector_store = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="./data/chromadb"
)

vector_store.persist()