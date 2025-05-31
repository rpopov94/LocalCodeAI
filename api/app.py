"""Main api."""
from fastapi import FastAPI
from core.rag.rag_core import RAGSystem
from core.llm.local_llm import LocalLLM

app = FastAPI()
rag = RAGSystem(r"C:\WorkSpace\websploit")
llm = LocalLLM()

@app.post("/build_knowledge_base")
async def build_kb():
    rag.build_knowledge_base()
    return {"status": "success"}

@app.post("/ask")
async def ask_question(question: str):
    context = rag.query(question)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    return {"answer": llm.generate(prompt)}
