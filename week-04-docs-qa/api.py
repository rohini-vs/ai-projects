"""HTTP API around answer.py.

    uvicorn api:app --reload
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "What is the deadline to submit an expense claim?"}'
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from answer import K, answer

app = FastAPI(title="Handbook Q&A")


class Ask(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=K, ge=1, le=10)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask):
    # plain `def` so FastAPI runs it in a worker thread (the LLM call blocks)
    return answer(body.question, k=body.k)