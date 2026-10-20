import sys

from blocks import load_blocks
from chunk_fixed import chunk_fixed
from chunk_recursive import chunk_recursive
from chunk_structure import chunk_structure
from db import connect, init_schema, insert_chunks
from embedder import embed_documents

PDF = "data/northwind-handbook.pdf"
STRATEGIES = {
    "fixed":     lambda b: chunk_fixed(b, 800, 100),
    "recursive": lambda b: chunk_recursive(b, 800, 100),
    "structure": lambda b: chunk_structure(b, 1200, 100),
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STRATEGIES)
    blocks = load_blocks(PDF)
    conn = connect()
    init_schema(conn)
    for name in wanted:
        chunks = STRATEGIES[name](blocks)
        vectors = embed_documents([c["text"] for c in chunks])
        insert_chunks(conn, name, chunks, vectors)
        print(f"{name:<10} {len(chunks)} chunks embedded and stored")