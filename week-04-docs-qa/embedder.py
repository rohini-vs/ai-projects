import httpx
import numpy as np

URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def embed(texts: list[str]) -> np.ndarray:
    r = httpx.post(URL, json={"model": MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def embed_documents(texts: list[str], batch: int = 16) -> np.ndarray:
    """Chunks get the 'search_document:' prefix, in small batches."""
    out = []
    for i in range(0, len(texts), batch):
        out.append(embed(["search_document: " + t for t in texts[i:i + batch]]))
    return np.vstack(out)


def embed_query(question: str) -> np.ndarray:
    """Questions get the 'search_query:' prefix (as in the probe)."""
    return embed(["search_query: " + question])[0]