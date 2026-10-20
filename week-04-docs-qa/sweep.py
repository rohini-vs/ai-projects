"""Try different chunk size / overlap settings and score each with the 16 questions.

Each config is stored in the same `chunks` table under its own label
(e.g. 'recursive-500-60'), so nothing you already ingested is touched.
"""
from blocks import load_blocks
from chunk_fixed import chunk_fixed
from chunk_recursive import chunk_recursive
from compare import QA, rank_of
from db import connect, init_schema, insert_chunks, search
from embedder import embed_documents, embed_query

PDF = "data/northwind-handbook.pdf"
K = 5

CONFIGS = [
    ("fixed", chunk_fixed, 300, 40),
    ("fixed", chunk_fixed, 800, 0),
    ("fixed", chunk_fixed, 800, 100),
    ("fixed", chunk_fixed, 1500, 180),
    ("recursive", chunk_recursive, 300, 40),
    ("recursive", chunk_recursive, 500, 60),
    ("recursive", chunk_recursive, 800, 0),
    ("recursive", chunk_recursive, 800, 100),
    ("recursive", chunk_recursive, 800, 200),
    ("recursive", chunk_recursive, 1500, 180),
]


if __name__ == "__main__":
    blocks = load_blocks(PDF)
    conn = connect()
    init_schema(conn)

    # embed the 16 questions once, reuse for every config
    qvecs = [embed_query(q) for q, *_ in QA]

    print(f"{'config':<20}{'chunks':>7}{'avg len':>9}{'top1':>6}{'top3':>6}{'top5':>6}")
    for name, fn, size, overlap in CONFIGS:
        label = f"{name}-{size}-{overlap}"
        chunks = fn(blocks, size=size, overlap=overlap)
        vectors = embed_documents([c["text"] for c in chunks])
        insert_chunks(conn, label, chunks, vectors)

        ranks = []
        for (q, trap, rule, phrases), qv in zip(QA, qvecs):
            texts = [h["text"] for h in search(conn, qv, label, K)]
            ranks.append(rank_of(texts, rule, phrases))

        avg = sum(len(c["text"]) for c in chunks) // len(chunks)
        counts = [sum(1 for r in ranks if r is not None and r <= k) for k in (1, 3, 5)]
        print(f"{label:<20}{len(chunks):>7}{avg:>9}{counts[0]:>6}{counts[1]:>6}{counts[2]:>6}", flush=True)

    print("\nSanity: fixed-800-100 should be 7/11/13, recursive-800-100 should be 9/13/15.")