# RAG PROCESSOR


```text
project_rag/
├── core/                     
│   ├── parser/               
│   │   ├── python_parser.py
│   │   ├── cpp_parser.py
│   │   └── ...
│   ├── embedding/            
│   │   ├── local_embeddings.py
│   │   └── ...
│   ├── vector_db/             
│   │   ├── chroma_manager.py
│   │   └── ...
│   └── rag/                   
│       └── rag_core.py
├── models/                    
│   ├── embedding/
│   └── llm/
├── configs/                  
│   ├── parser_config.yaml
│   └── embedding_config.yaml
├── api/                       
│   └── fastapi_app.py
└── main.py                    
```