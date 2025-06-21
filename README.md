# RAG Processor Backend

A backend service for working with RAG (Retrieval-Augmented Generation), running entirely locally on the CPU.

## Features:

- Download documents from a folder
- Text vectorization via Sentence Transformers
- Local generation model (for example TinyLlama)
- REST API for integration

## Installation:

```bash
uv pip install -r requirements.txt
```

The project is a local RAG system consisting of several key components:
```text
rag_processor/
├── core/
│   ├── embeddings/
│   │   └── local_ebeddings.py # Vector text repr
│   ├── llm/
│   │   └── local_llm.py       # local language model
│   ├── parser/
│   │   └──docs.py             # docs loader
│   ├── rag/
│   │   └── rag_core.py        # core of rag system
│   └── vector_db/
│       └── chroma_manager.py  # vector database
├── .gitignore
├── .python-version  
├── app.py                      # Fastapi application   
├── config.py                   # App config
├── pyproject.toml              # uv depencies
├── README.md
└── uv.lock                                                      
```

## Running

```bash
uv run python app.py
```
or
```bash
uv run start
```


## Api interface
FastAPI application with two endpoints:

- /build_knownge_base - building knowledge base

- /ask - a question and answer using RAG
