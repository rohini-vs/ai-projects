import typer
from rich.console import Console
from rich.markup import escape

from agent import solve

app = typer.Typer()
console = Console()

QUESTIONS = {
    "single_tool": "What is 6847 times 3928?",
    "multi_step": "How many words are in the sentence 'the quick brown fox jumps over the lazy dog', and what is that number multiplied by 12?",
}


@app.command()
def run(model: str = typer.Option(...)):
    for label, question in QUESTIONS.items():
        console.print(f"\n[bold]{label}[/bold]: {question}")
        answer = solve(question, model, verbose=True)
        console.print(f"[dim]Final answer:[/dim] {escape(answer)}")


if __name__ == "__main__":
    app()