"""Main api."""
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query, HTTPException

from config import DOC_SOURSES, MODEL_PATH
from core.rag.rag_core import RAGSystem
from core.llm.local_llm import LocalLLM


rag = RAGSystem(DOC_SOURSES)
llm = LocalLLM(MODEL_PATH)

app = FastAPI()

@app.get("/build_knowledge_base")
async def build_kb():
    rag.build_knowledge_base()
    return {"status": "success"}


@app.get("/ask")
async def ask_question(
    question: str = Query(..., min_length=3, description="Your question"),
    top_k: Optional[int] = Query(3, gt=0, le=10, description="Number of results returned")
):
    try:
        context = rag.query(question, top_k=top_k)

        prompt = f"""Context: {context}

        Question: {question}

        Answer:"""

        answer = llm.generate(prompt)

        return {
            "question": question,
            "context": context,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
