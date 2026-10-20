import pymupdf

def extract_pages(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    return [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]

if __name__ == "__main__":
    pages = extract_pages("data/northwind-handbook.pdf")
    print("pages:", len(pages))
    print("chars per page:", [len(p["text"]) for p in pages])
    print(repr(pages[6]["text"][:450]))   # page 7 = expenses