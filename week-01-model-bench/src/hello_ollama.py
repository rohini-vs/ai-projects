import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    data = generate("llama3.2:3b", "Explain recursion to a 10-year-old in 3 sentences.")

    ns = 1e9  # Ollama reports nanoseconds
    prefill_tps = data["prompt_eval_count"] / (data["prompt_eval_duration"] / ns)
    decode_tps = data["eval_count"] / (data["eval_duration"] / ns)

    print(data["response"])
    print("-" * 40)
    print(f"Load time:     {data['load_duration'] / ns:.2f}s")
    print(f"Prompt tokens: {data['prompt_eval_count']}  ({prefill_tps:.1f} tok/s prefill)")
    print(f"Output tokens: {data['eval_count']}  ({decode_tps:.1f} tok/s decode)")
    print(f"Total time:    {data['total_duration'] / ns:.2f}s")