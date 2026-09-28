import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from prompts import PROMPTS

app = typer.Typer()
console = Console()

EXPECTED = {p["id"]: p.get("expected") for p in PROMPTS}

JSON_EXPECTED = {
    "json-1": {"name": "Priya", "age": 29, "city": "Chennai"},
    "json-2": {"product": "ThinkPad X1", "price": 1499, "currency": "USD"},
}

CODE_TESTS = {
    "code-1": (
        "assert is_palindrome('A man, a plan, a canal: Panama')\n"
        "assert not is_palindrome('hello')\n"
        "assert is_palindrome('No lemon, no melon')\n"
    ),
    "code-2": "assert fib(10) == 55\nassert fib(20) == 6765\n",
}

FENCE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)


def strip_fences(text: str) -> str:
    m = FENCE.search(text)
    return (m.group(1) if m else text).strip()


def score_text(row) -> bool:
    answer = row.response.strip().lower()
    return re.search(rf"\b{re.escape(EXPECTED[row.id])}\b", answer) is not None


def score_json(row) -> bool:
    try:
        data = json.loads(strip_fences(row.response))
    except json.JSONDecodeError:
        return False
    return data == JSON_EXPECTED[row.id]  # "1499" != 1499, so types matter


def score_code(row) -> bool:
    program = strip_fences(row.response) + "\n\n" + CODE_TESTS[row.id]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


SCORERS = {"factual": score_text, "reasoning": score_text,
           "coding": score_code, "json": score_json}


@app.command()
def main(path: Path | None = typer.Argument(None, help="Results CSV (default: latest in results/)")):
    path = path or sorted(Path("results").glob("bench_*.csv"))[-1]
    df = pd.read_csv(path)
    df["response"] = df.response.fillna("")

    scored = df[df.category.isin(SCORERS)].copy()
    scored["correct"] = scored.apply(lambda r: SCORERS[r.category](r), axis=1)
    scored["fenced"] = scored.response.str.contains("```")

    cats = ["factual", "reasoning", "coding", "json"]
    table = Table(title=f"Quality: {path.name}")
    for c in ["Model", *[c.title() for c in cats], "Total", "Fenced JSON"]:
        table.add_column(c, justify="left" if c == "Model" else "right")
    for model, g in scored.groupby("model"):
        per_cat = [f"{int(g[g.category == c].correct.sum())}/{(g.category == c).sum()}" for c in cats]
        fenced = int(g[g.category == "json"].fenced.sum())
        table.add_row(model, *per_cat, f"{int(g.correct.sum())}/{len(g)}", str(fenced))
    console.print(table)

    console.print("\n[bold]Failures[/bold]")
    for _, r in scored[~scored.correct].iterrows():
        console.print(f"[red]✗[/red] {r.model} {r.id}: {escape(repr(r.response.strip()[:100]))}")

    console.print("\n[bold]Summaries: judge these yourself[/bold]")
    for _, r in df[df.category == "summarization"].iterrows():
        console.print(f"[cyan]{r.model} / {r.id}[/cyan]: {escape(r.response.strip())}\n")

    out = path.with_name(f"scored_{path.name}")
    scored.to_csv(out, index=False)
    console.print(f"Saved [green]{out}[/green]")


if __name__ == "__main__":
    app()