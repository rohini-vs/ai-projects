import typer
from rich.console import Console
from rich.table import Table

from ollama_client import count_tokens, unload

app = typer.Typer()
console = Console()

# Same meaning in both languages
PAIRS = [
    ("The weather is very hot today.",
     "இன்று வானிலை மிகவும் சூடாக இருக்கிறது."),
    ("I am going to the market to buy vegetables.",
     "நான் காய்கறிகள் வாங்க சந்தைக்குச் செல்கிறேன்."),
    ("Chennai is the capital of Tamil Nadu.",
     "சென்னை தமிழ்நாட்டின் தலைநகரம்."),
    ("Please send me the report by tomorrow morning.",
     "தயவுசெய்து அறிக்கையை நாளை காலைக்குள் எனக்கு அனுப்புங்கள்."),
    ("Learning new skills takes patience and regular practice.",
     "புதிய திறன்களைக் கற்றுக்கொள்ள பொறுமையும் தொடர்ச்சியான பயிற்சியும் தேவை."),
    ("Artificial intelligence is changing how software is built.",
     "செயற்கை நுண்ணறிவு மென்பொருள் உருவாக்கப்படும் விதத்தை மாற்றிக்கொண்டிருக்கிறது."),
]


@app.command()
def run(models: str = typer.Option(..., help="Comma-separated model names")):
    table = Table(title="Token cost: same 6 sentences, English vs Tamil")
    for col in ["Model", "English tokens", "Tamil tokens", "Tamil costs"]:
        table.add_column(col, justify="left" if col == "Model" else "right")

    for model in [m.strip() for m in models.split(",")]:
        unload(model)  # start cold so Ollama's prompt cache can't distort counts
        en = sum(count_tokens(model, e) for e, _ in PAIRS)
        ta = sum(count_tokens(model, t) for _, t in PAIRS)
        table.add_row(model, str(en), str(ta), f"{ta / en:.1f}x")
        unload(model)

    console.print(table)


if __name__ == "__main__":
    app()