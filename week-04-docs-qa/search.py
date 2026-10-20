import argparse

from db import connect, search
from embedder import embed_query

STRATEGIES = ["fixed", "recursive", "structure"]


def show(name: str, hits: list[dict], full: bool):
    print(f"\n=== {name} ===")
    for i, h in enumerate(hits, 1):
        head = f" | {h['heading']}" if h["heading"] else ""
        print(f"{i}. score {h['score']:.3f}  p.{h['page']}{head}")
        text = h["text"] if full else h["text"].replace("\n", " ")[:160] + "..."
        print("   " + text.replace("\n", "\n   "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-s", "--strategy", choices=STRATEGIES + ["all"], default="all")
    ap.add_argument("-k", type=int, default=5, help="how many chunks to return")
    ap.add_argument("--full", action="store_true", help="print whole chunks")
    a = ap.parse_args()

    conn = connect()
    qvec = embed_query(a.question)          # question → 768 numbers
    for name in (STRATEGIES if a.strategy == "all" else [a.strategy]):
        show(name, search(conn, qvec, name, a.k), a.full)