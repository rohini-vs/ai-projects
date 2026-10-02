import typer
from rich.console import Console
from rich.table import Table

from extract import extract

app = typer.Typer()
console = Console()

POSTINGS = {
    "easy": """New job alert - Backend Engineer needed at Flexon Systems, Chennai office.
3-5 yrs exp required. Pay range 12-18 LPA depending on experience.
Remote not available, hybrid 3 days/week.""",
    "hard": """We're hiring! Senior Data Analyst role, open to remote for the right candidate but prefer Bangalore-based folks.
Looking for someone with 4+ years, ideally up to 7. Comp is around ₹15-22 LPA, negotiable for the right fit.
Company: Verlix Analytics.""",
}


@app.command()
def run(models: str = typer.Option(...)):
    table = Table(title="Retry loop: attempts needed per model/posting")
    table.add_column("Model")
    table.add_column("Posting")
    table.add_column("Result")
    table.add_column("Attempts", justify="right")

    for model in [m.strip() for m in models.split(",")]:
        for name, text in POSTINGS.items():
            job, attempts = extract(model, text)
            status = "[green]✓[/green]" if job else "[red]✗ gave up[/red]"
            table.add_row(model, name, status, str(attempts))

    console.print(table)


if __name__ == "__main__":
    app()