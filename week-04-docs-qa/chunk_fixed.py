import bisect

from blocks import load_blocks


def chunk_fixed(blocks: list[dict], size: int = 800, overlap: int = 100) -> list[dict]:
    """Naive chunker: glue all text together, then cut every `size` characters.

    Each window starts `size - overlap` characters after the previous one,
    so neighbouring chunks share `overlap` characters.
    Knows nothing about sentences, headings or tables.
    """
    assert 0 <= overlap < size

    # 1. one long string, remembering where each block starts (to recover the page later)
    text, starts, pages = "", [], []
    for b in blocks:
        starts.append(len(text))
        pages.append(b["page"])
        text += b["text"] + "\n"

    # 2. slide a window over it
    chunks, step = [], size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if not piece:
            continue
        # page of the block that contains the first character of this window
        page = pages[bisect.bisect_right(starts, start) - 1]
        chunks.append({"id": len(chunks), "page": page, "text": piece})
        if start + size >= len(text):      # window already reached the end
            break
    return chunks


if __name__ == "__main__":
    blocks = load_blocks("data/northwind-handbook.pdf")
    chunks = chunk_fixed(blocks, size=800, overlap=100)

    lens = [len(c["text"]) for c in chunks]
    print(f"chunks: {len(chunks)}  | chars min/avg/max: {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

    print("\nchunk 12 (page %d):" % chunks[12]["page"])
    print(chunks[12]["text"])

    print("\n--- the seam between chunk 12 and 13 ---")
    print("END of 12  :", repr(chunks[12]["text"][-120:]))
    print("START of 13:", repr(chunks[13]["text"][:120]))