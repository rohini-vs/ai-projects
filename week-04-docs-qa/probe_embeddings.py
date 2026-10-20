import httpx
import numpy as np

def embed(texts: list[str]) -> np.ndarray:
    r = httpx.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": texts},
        timeout=60,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"])

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

docs = [
    "Customers can return items within 30 days for a full refund.",
    "The server restarts automatically every Sunday at 2 AM.",
    "Our refund policy allows money back if you return a product in a month.",
]
query = "How do I get my money back?"

# nomic-embed-text expects these task prefixes
doc_vecs = embed([f"search_document: {d}" for d in docs])
q_vec = embed([f"search_query: {query}"])[0]

print("vector shape:", doc_vecs.shape)
for d, v in zip(docs, doc_vecs):
    print(f"{cosine(q_vec, v):.3f}  {d}")