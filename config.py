"""Project config."""
import os

from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    VECTOR_DB_PATH = os.path.join(DATA_DIR, "vectorstore")
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_NAME = os.getenv('LLM_NAME')
    HUGGINGFACE = os.getenv('HUGGINGFACEHUB_API_TOKEN')

config = Config()
