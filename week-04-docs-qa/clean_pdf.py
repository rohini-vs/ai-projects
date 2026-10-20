import re
from collections import Counter

from extract_pdf import extract_pages


def normalize(line: str) -> str:
    """Mask digits so 'Page 7' and 'Page 12' count as the same line."""
    return re.sub(r"\d+", "#", line.strip())


def find_boilerplate(pages: list[dict], min_share: float = 0.5) -> set[str]:
    """Lines (digits masked) that appear on at least min_share of all pages."""
    counts = Counter()
    for p in pages:
        # a set, so a line is counted once per page however often it repeats
        counts.update({normalize(l) for l in p["text"].splitlines() if l.strip()})
    cutoff = len(pages) * min_share
    return {line for line, n in counts.items() if n >= cutoff}


def remove_boilerplate(pages: list[dict]) -> tuple[list[dict], set[str]]:
    boiler = find_boilerplate(pages)
    cleaned = []
    for p in pages:
        kept = [l for l in p["text"].splitlines() if normalize(l) not in boiler]
        cleaned.append({"page": p["page"], "text": "\n".join(kept)})
    return cleaned, boiler


if __name__ == "__main__":
    pages = extract_pages("data/northwind-handbook.pdf")
    cleaned, boiler = remove_boilerplate(pages)

    print("boilerplate lines found:")
    for b in sorted(boiler):
        print("  ", repr(b))

    removed = [len(p["text"].splitlines()) - len(c["text"].splitlines())
               for p, c in zip(pages, cleaned)]
    print("lines removed per page:", removed)
    print("chars per page after:", [len(c["text"]) for c in cleaned])
    print()
    print("page 7 BEFORE:", repr(pages[6]["text"][:200]))
    print("page 7 AFTER: ", repr(cleaned[6]["text"][:200]))