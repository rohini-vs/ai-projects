from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ollama_client import NS, generate, running_models, unload
from prompts import PROMPTS

app = typer.Typer()
console = Console()


def model_memory_gb(model: str) -> float | None:
    for m in running_models():
        if m["name"] == model:
            return m["size"] / 1e9
    return None


@app.command()
def run(
    models: str = typer.Option(..., help="Comma-separated model names, e.g. llama3.2:3b,qwen2.5:7b"),
    temperature: float = 0.0,
    max_tokens: int = 256,
):
    rows = []
    memory = {}

    for model in [m.strip() for m in models.split(",")]:
        console.print(f"\n[bold]{model}[/bold]: warm-up (loading into memory)")
        generate(model, "Say hi.", max_tokens=8)  # untimed: pays the load cost once

        for p in track(PROMPTS, description=f"Benchmarking {model}"):
            d = generate(model, p["prompt"], temperature, max_tokens)
            rows.append({
                "model": model,
                "id": p["id"],
                "category": p["category"],
                "prompt_tokens": d["prompt_eval_count"],
                "output_tokens": d["eval_count"],
                "prefill_s": d["prompt_eval_duration"] / NS,
                "decode_s": d["eval_duration"] / NS,
                "total_s": d["total_duration"] / NS,
                "response": d["response"],
            })

        memory[model] = model_memory_gb(model)
        unload(model)  # free RAM before the next model

    df = pd.DataFrame(rows)

    # Save raw results
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"bench_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)

    # Aggregate: totals first, then divide (a proper weighted average)
    g = df.groupby("model").agg(
        prompt_tokens=("prompt_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        prefill_s=("prefill_s", "sum"),
        decode_s=("decode_s", "sum"),
        total_s=("total_s", "sum"),
    )
    g["prefill_tps"] = g.prompt_tokens / g.prefill_s
    g["decode_tps"] = g.output_tokens / g.decode_s

    table = Table(title="Local Model Bench")
    for col in ["Model", "Memory (GB)", "Prefill tok/s", "Decode tok/s", "Total time (s)"]:
        table.add_column(col, justify="left" if col == "Model" else "right")
    for model, r in g.iterrows():
        mem = memory.get(model)
        table.add_row(model, f"{mem:.1f}" if mem else "?",
                      f"{r.prefill_tps:.0f}", f"{r.decode_tps:.1f}", f"{r.total_s:.1f}")
    console.print(table)
    console.print(f"Raw results saved to [green]{out}[/green]")


if __name__ == "__main__":
    app()