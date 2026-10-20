from blocks import load_blocks
from chunk_recursive import chunk_recursive


def split_sections(blocks: list[dict]) -> list[dict]:
    """Group blocks into sections. A new section starts at every h2 heading
    (and at every h1). Each section remembers its heading path, e.g.
    '4. Expense claims > 4.4 Approval of claims'."""
    sections, chapter, current = [], "", None
    for b in blocks:
        if b["kind"] == "h1":
            chapter = b["text"]
            current = None                       # chapter title alone has no body
        elif b["kind"] == "h2":
            current = {"path": f"{chapter} > {b['text']}" if chapter else b["text"],
                       "title": b["text"], "body": []}
            sections.append(current)
        else:                                    # body text
            if current is None:                  # text directly under a chapter title
                current = {"path": chapter, "title": chapter, "body": []}
                sections.append(current)
            current["body"].append(b)
    return [s for s in sections if s["body"]]


def chunk_structure(blocks: list[dict], max_size: int = 1200, overlap: int = 100) -> list[dict]:
    """One chunk per section. A section longer than max_size is cut with the
    recursive chunker, and every piece keeps the heading path on top."""
    chunks = []
    for s in split_sections(blocks):
        header = s["path"] + "\n"
        body = "\n".join(b["text"] for b in s["body"])
        p0, p1 = s["body"][0]["page"], s["body"][-1]["page"]

        if len(header) + len(body) <= max_size:
            chunks.append({"id": len(chunks), "page": p0, "page_end": p1,
                           "heading": s["path"], "text": header + body})
        else:
            for piece in chunk_recursive(s["body"], size=max_size - len(header), overlap=overlap):
                chunks.append({"id": len(chunks), "page": piece["page"], "page_end": piece["page_end"],
                               "heading": s["path"], "text": header + piece["text"]})
    return chunks


if __name__ == "__main__":
    blocks = load_blocks("data/northwind-handbook.pdf")
    chunks = chunk_structure(blocks)

    lens = [len(c["text"]) for c in chunks]
    print(f"chunks: {len(chunks)}  | chars min/avg/max: {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

    for c in chunks:
        if "4.3" in c["heading"]:
            print(f"\n--- chunk {c['id']} | page {c['page']} ---")
            print(c["text"])
            break