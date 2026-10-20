import bisect

from blocks import load_blocks

# biggest natural boundary first; the last resort (a hard cut) is handled in code
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _atoms(text: str, start: int, end: int, seps: list[str], size: int) -> list[tuple[int, int]]:
    """Split text[start:end] into contiguous spans, each <= size where possible.

    Try the biggest separator first. A span that is still too big is split again
    with the next separator. Only if no separator is left do we cut blindly.
    """
    if end - start <= size:
        return [(start, end)]
    if not seps:                                   # last resort: hard cut
        return [(i, min(i + size, end)) for i in range(start, end, size)]

    sep, rest = seps[0], seps[1:]
    spans, pos = [], start
    while pos < end:
        i = text.find(sep, pos, end)
        cut = end if i == -1 else i + len(sep)     # keep the separator on the left piece
        spans.append((pos, cut))
        pos = cut

    out = []
    for a, b in spans:
        out.extend(_atoms(text, a, b, rest, size) if b - a > size else [(a, b)])
    return out


def chunk_recursive(blocks: list[dict], size: int = 800, overlap: int = 100) -> list[dict]:
    # one string; a blank line between blocks so "\n\n" means "block boundary"
    text, starts, pages = "", [], []
    for b in blocks:
        starts.append(len(text))
        pages.append(b["page"])
        text += b["text"] + "\n\n"

    atoms = _atoms(text, 0, len(text), SEPARATORS, size)

    # glue neighbouring atoms back together until the chunk would exceed `size`
    chunks, i, n = [], 0, len(atoms)
    while i < n:
        j = i
        while j < n and atoms[j][1] - atoms[i][0] <= size:
            j += 1
        j = max(j, i + 1)
        a, b = atoms[i][0], atoms[j - 1][1]

        piece = text[a:b].strip()
        if piece:
            first = a + (len(text[a:b]) - len(text[a:b].lstrip()))
            last = b - (len(text[a:b]) - len(text[a:b].rstrip())) - 1
            page = lambda pos: pages[bisect.bisect_right(starts, pos) - 1]
            chunks.append({"id": len(chunks), "page": page(first),
                           "page_end": page(last), "text": piece})
        if j >= n:
            break
        # next chunk starts early enough to repeat <= `overlap` characters,
        # but always on an atom boundary (never mid-sentence)
        k = j
        while k - 1 > i and b - atoms[k - 1][0] <= overlap:
            k -= 1
        i = k
    return chunks


if __name__ == "__main__":
    blocks = load_blocks("data/northwind-handbook.pdf")
    chunks = chunk_recursive(blocks, size=800, overlap=100)

    lens = [len(c["text"]) for c in chunks]
    print(f"chunks: {len(chunks)}  | chars min/avg/max: {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

    print("\nchunk 12 (page %d):" % chunks[12]["page"])
    print(chunks[12]["text"])

    print("\n--- the seam between chunk 12 and 13 ---")
    print("END of 12  :", repr(chunks[12]["text"][-120:]))
    print("START of 13:", repr(chunks[13]["text"][:120]))