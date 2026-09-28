from pathlib import Path

from rich.console import Console
from rich.table import Table

from tiny_bpe import encode, train

console = Console()
VOCAB_SIZE = 512  # 256 raw bytes + 256 learned merges


def load(name: str) -> tuple[str, str]:
    text = Path("data", name).read_text(encoding="utf-8")
    cut = int(len(text) * 0.8)
    return text[:cut], text[cut:]  # train part, held-out test part


def tokens_per_char(text: str, merges: dict) -> float:
    return len(encode(text, merges)) / len(text)


def main():
    en_train, en_test = load("english.txt")
    ta_train, ta_test = load("tamil.txt")

    console.print("Training tokenizers (this takes a little while)...")
    tokenizers = {
        "no merges (raw bytes)": {},
        "English-only": train(en_train, VOCAB_SIZE),
        "Tamil-only": train(ta_train, VOCAB_SIZE),
        "English + Tamil": train(en_train + "\n" + ta_train, VOCAB_SIZE),
    }

    table = Table(title="Tokens per character on held-out text (lower is better)")
    table.add_column("Tokenizer (trained on)")
    table.add_column("English", justify="right")
    table.add_column("Tamil", justify="right")
    for name, merges in tokenizers.items():
        table.add_row(name, f"{tokens_per_char(en_test, merges):.2f}",
                      f"{tokens_per_char(ta_test, merges):.2f}")
    console.print(table)


if __name__ == "__main__":
    main()