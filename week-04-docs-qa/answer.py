"""Ask a question -> retrieve top chunks -> LLM answers using only those chunks, with citations.

    python answer.py "What is the deadline to submit an expense claim?"
    python answer.py --all          # run all 16 test questions and print the answers
"""
import argparse
import json
import urllib.request

from db import connect, search
from embedder import embed_query

STRATEGY = "recursive-800-100"      # the chunk set chosen from the sweep
K = 5
MODEL = "qwen2.5:7b"
CHAT_URL = "http://localhost:11434/api/chat"

SYSTEM = """You answer questions about the Northwind Labs handbook.

Rules:
1. Use ONLY the sources provided. Never use outside knowledge.
2. Cite every fact with its source tag, like [S1] or [S2].
3. If a later notice (for example in Appendix A) changes a rule in the main text, the notice wins. Say what the main text says AND what the notice changed.
4. If the sources do not contain the answer, reply exactly: "I couldn't find this in the handbook." Do not guess. If the sources only point elsewhere (for example "see your offer letter"), say that.
5. Be short and direct."""


def build_prompt(question: str, hits: list[dict]) -> str:
    sources = "\n\n".join(
        f"[S{i}] (page {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    return f"SOURCES:\n\n{sources}\n\nQUESTION: {question}"


def ask_llm(prompt: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(CHAT_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"].strip()


def answer(question: str, conn=None, k: int = K) -> dict:
    """Returns {'answer': str, 'sources': [{'tag','page','score','text'}]}."""
    conn = conn or connect()
    hits = search(conn, embed_query(question), STRATEGY, k)
    text = ask_llm(build_prompt(question, hits))
    sources = [{"tag": f"S{i}", "page": h["page"], "score": round(h["score"], 3), "text": h["text"]}
               for i, h in enumerate(hits, 1)]
    return {"answer": text, "sources": sources}


def show(question: str, result: dict) -> None:
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print("   sources: " + ", ".join(
        f"{s['tag']}=p.{s['page']} ({s['score']})" for s in result["sources"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--all", action="store_true", help="run the 16 test questions")
    args = ap.parse_args()

    conn = connect()
    if args.all:
        from compare import QA
        for q, *_ in QA:
            show(q, answer(q, conn))
    elif args.question:
        show(args.question, answer(args.question, conn))
    else:
        ap.error("give a question or --all")