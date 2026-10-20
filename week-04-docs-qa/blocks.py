from collections import Counter

import pymupdf

from clean_pdf import normalize

SKIP_PAGES = {1, 2}  # cover and table of contents


def extract_blocks(pdf_path: str) -> list[dict]:
    """One dict per text block (a paragraph, a heading, a table cell...), in reading order."""
    doc = pymupdf.open(pdf_path)
    blocks = []
    for pno, page in enumerate(doc, 1):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:          # 0 = text, 1 = image
                continue
            lines = ["".join(s["text"] for s in l["spans"]).strip() for l in b["lines"]]
            text = " ".join(l for l in lines if l)   # unwrap: join the visual lines
            if not text:
                continue
            spans = [s for l in b["lines"] for s in l["spans"]]
            size = round(max(spans, key=lambda s: len(s["text"]))["size"], 1)
            blocks.append({"page": pno, "text": text, "size": size,
                           "x0": b["bbox"][0], "y0": b["bbox"][1]})
    # reading order: page, then top to bottom, then left to right
    blocks.sort(key=lambda b: (b["page"], round(b["y0"]), b["x0"]))
    return blocks


def drop_repeated(blocks: list[dict], min_share: float = 0.5) -> list[dict]:
    """Same idea as clean_pdf, but per block: drop blocks repeated on >= half the pages."""
    n_pages = len({b["page"] for b in blocks})
    pages_seen = Counter()
    for page, key in {(b["page"], normalize(b["text"])) for b in blocks}:
        pages_seen[key] += 1
    return [b for b in blocks if pages_seen[normalize(b["text"])] < n_pages * min_share]


def label(blocks: list[dict]) -> None:
    """Mark each block h1 / h2 / text from its font size."""
    chars = Counter()
    for b in blocks:
        chars[b["size"]] += len(b["text"])
    body = chars.most_common(1)[0][0]               # most common size = body text
    heads = sorted({b["size"] for b in blocks if b["size"] >= body + 1.5}, reverse=True)
    for b in blocks:
        b["kind"] = f"h{heads.index(b['size']) + 1}" if b["size"] in heads else "text"


def load_blocks(pdf_path: str) -> list[dict]:
    blocks = drop_repeated(extract_blocks(pdf_path))
    blocks = [b for b in blocks if b["page"] not in SKIP_PAGES]
    label(blocks)
    return blocks


if __name__ == "__main__":
    blocks = load_blocks("data/northwind-handbook.pdf")
    print("blocks:", len(blocks), dict(Counter(b["kind"] for b in blocks)))
    print("\nchapters:")
    for b in blocks:
        if b["kind"] == "h1":
            print(f"  p{b['page']:<3}{b['text']}")
    print("\npage 7 in reading order:")
    for b in blocks:
        if b["page"] == 7:
            print(f"  {b['kind']:<5}{b['text'][:80]!r}")