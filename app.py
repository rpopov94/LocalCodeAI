"""Main api."""
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from core.rag.rag_core import RAGSystem
from core.llm.local_llm import LocalLLM


rag = RAGSystem(r"C:\WorkSpace\websploit")
llm = LocalLLM()

app = FastAPI()

@app.get("/build_knowledge_base")
async def build_kb():
    rag.build_knowledge_base()
    return {"status": "success"}


@app.get("/ask")
async def ask_question(
        question: str = Query(..., min_length=3, description="Ваш вопрос для ИИ"),
        top_k: Optional[int] = Query(3, gt=0, le=10, description="Количество возвращаемых результатов")
):
    try:
        context = rag.query(question, top_k=top_k)

        prompt = f"""Контекст: {context}

        Вопрос: {question}

        Ответ:"""

        answer = llm.generate(prompt)

        return {
            "question": question,
            "context": context,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)