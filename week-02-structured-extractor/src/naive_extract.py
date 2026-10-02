import json

import typer
from rich.console import Console

from ollama_client import generate
from schema import JobPosting

app = typer.Typer()
console = Console()

JOB_TEXT = """We're hiring! Senior Data Analyst role, open to remote for the right candidate but prefer Bangalore-based folks.
Looking for someone with 4+ years, ideally up to 7. Comp is around ₹15-22 LPA, negotiable for the right fit.
Company: Verlix Analytics."""

PROMPT = f"""Extract the job details from this posting as a JSON object with these exact keys:
title, company, location, min_experience_years, max_experience_years, salary_min_lpa, salary_max_lpa, remote.

Posting:
{JOB_TEXT}

Respond with only the JSON object, nothing else."""

# qwen3:4b was tested and retired from this week's lineup. It is a hybrid
# reasoning model that "thinks" before answering, and in this Ollama setup
# that thinking overhead proved unbounded rather than fixed:
#
#   Budget   Thinking used   Result
#   300      300 (maxed)     cut off, no output
#   1200     ~1180 (maxed)   cut off, no output
#   3000     ~3000 (maxed)   cut off, no output (even on an EASY posting)
#   4500     ~4485 (maxed)   cut off, no output
#
# Three documented ways to disable thinking under Ollama were also tried and
# none worked:
#   1. API-level `think=False` flag       -> known gap in Ollama's Qwen3 support
#   2. Inline "/no_think" soft-switch     -> Ollama's packaged template doesn't
#                                             appear to detect the phrase
#   3. Manual empty <think></think> tags
#      in a raw-mode prompt               -> raw mode likely sends special-token
#                                             strings as literal text, not real
#                                             control tokens, so no turn boundary
#                                             was ever recognized
#
# Conclusion: qwen3:4b is not currently usable for simple structured extraction
# under Ollama without unbounded thinking overhead. Excluded from the rest of
# this week's comparison.
TOKEN_BUDGETS = {
    "qwen3:4b": 4500,
}
DEFAULT_BUDGET = 300


@app.command()
def run(model: str = typer.Option(...)):
    max_tokens = TOKEN_BUDGETS.get(model, DEFAULT_BUDGET)
    result = generate(model, PROMPT, temperature=0.0, max_tokens=max_tokens)

    console.print(
        f"[dim]done_reason: {result.get('done_reason')} | "
        f"eval_count: {result.get('eval_count')} | "
        f"budget: {max_tokens}[/dim]"
    )
    if result.get("thinking"):
        console.print(f"[dim]Thinking length: {len(result['thinking'])} chars[/dim]")

    raw = result["response"]
    console.print(f"[bold]Raw output from {model}:[/bold]")
    console.print(raw)
    console.print()

    try:
        data = json.loads(raw)
        job = JobPosting(**data)
        console.print("[green]✓ Parsed and validated successfully[/green]")
        console.print(job)
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")


if __name__ == "__main__":
    app()