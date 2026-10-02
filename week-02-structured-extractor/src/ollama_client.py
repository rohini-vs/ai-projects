import httpx

BASE_URL = "http://localhost:11434"
NS = 1e9  # Ollama reports durations in nanoseconds


def generate(model: str, prompt: str, temperature: float = 0.0,
             max_tokens: int = 256, keep_alive: str = "5m",
             think: bool | None = None, raw: bool = False) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
    }
    if think is not None:
        payload["think"] = think
    if raw:
        payload["raw"] = True
    resp = httpx.post(f"{BASE_URL}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def running_models() -> list[dict]:
    """Models currently loaded in memory, with their sizes."""
    resp = httpx.get(f"{BASE_URL}/api/ps", timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


def unload(model: str) -> None:
    """keep_alive=0 tells Ollama to evict the model from memory now."""
    httpx.post(f"{BASE_URL}/api/generate",
               json={"model": model, "keep_alive": 0}, timeout=60)


def count_tokens(model: str, text: str) -> int:
    """How many tokens does this model's tokenizer make from `text`?"""
    payload = {"model": model, "prompt": text, "raw": True, "stream": False,
               "options": {"num_predict": 1}}
    resp = httpx.post(f"{BASE_URL}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["prompt_eval_count"]