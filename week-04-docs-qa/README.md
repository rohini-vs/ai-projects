# Week 4: Docs Q&A (naive RAG)

Ask questions about a PDF in plain English and get a short answer with citations (page numbers), using only local models. No cloud APIs, no frameworks.

```
You:    What is the deadline to submit an expense claim?

Answer: The claim submission window is extended from 30 days to 45 days
        from the date of the expense due to Notice 2026-03 [S2].

Sources: [S1] page 7  (similarity 0.755)   <- Section 4.3: "within 30 days..."
         [S2] page 18 (similarity 0.732)   <- Notice 2026-03: "extended to 45 days"
         [S3] page 18 ...  [S4] page 7 ...  [S5] page 17 ...
```

That is a real output from this project. Note what happened: the main handbook text says 30 days, a later notice in an appendix changed it to 45, and the system found both and answered correctly.

---

## What is RAG, in one minute

A language model does not know your document. You could paste the whole document into the prompt, but long documents are slow, expensive, and the model gets distracted. **RAG (Retrieval-Augmented Generation)** does this instead:

1. **Retrieve**: find the few passages of the document most likely to contain the answer.
2. **Generate**: give only those passages to the model and ask it to answer from them.

This project builds both halves from scratch to see exactly what each step does and where it breaks.

---

## How it works

```
INGEST (run once per document)

  PDF
   │  PyMuPDF: text + font sizes + positions
   ▼
  clean         remove repeated headers / footers, skip cover + contents page
   │
   ▼
  blocks        paragraphs and headings, in reading order, with page numbers
   │
   ▼
  chunks        cut into ~800-character pieces (three strategies compared)
   │
   ▼
  embeddings    each chunk -> 768 numbers that capture its meaning
   │            (nomic-embed-text, running in Ollama)
   ▼
  Postgres + pgvector


QUERY (every question)

  question ──embed──► 768 numbers
                         │
                         ▼   cosine distance, nearest 5
                    top-5 chunks (with page numbers)
                         │
                         ▼   prompt: "answer only from these sources, cite them"
                    qwen2.5:7b (Ollama)
                         │
                         ▼
                    answer + citations
```

### The ideas in plain words

- **Embedding**: a list of numbers (768 here) that represents what a text *means*. Two texts with similar meaning get similar numbers, even with different words. "Money back for a cab" lands near "cab fares are reimbursable".
- **Cosine similarity**: how close two embeddings point. The score shown in the app is `1 - cosine distance`, so higher is closer. Scores sit in a narrow band (unrelated text still scores around 0.5), so the **ranking** matters, not the raw number.
- **Chunk**: the unit we embed, search and paste into the prompt. One vector for the whole PDF would be a blurry average, so we cut it up. Too small loses context; too big blurs the meaning.
- **Overlap**: the next chunk repeats the end of the previous one so an answer that straddles a cut is not lost.
- **Top-k**: how many chunks to retrieve (we use 5).
- **pgvector**: a Postgres extension that stores vectors and finds the nearest ones. We use an exact scan (no index) because we have a few hundred rows. At millions of rows you would add an approximate index (HNSW).
- **Prefixes**: `nomic-embed-text` expects `search_document: ` before stored text and `search_query: ` before questions. Skipping them quietly lowers quality.

---

## The three chunkers

All three are kept side by side in the same database table so they can be compared on the same questions.

| Strategy | How it cuts | Good | Bad |
|---|---|---|---|
| **fixed** | every 800 characters, 100 overlap | trivial to build | cuts mid-word, mixes topics in one chunk |
| **recursive** | tries to cut at paragraph, then line, then sentence, then word boundaries, up to 800 characters | never cuts mid-sentence, simple | still ignores sections: a chunk can start in one section and end in another; can leave a heading stranded at the end of a chunk |
| **structure** | one chunk per section, with the heading path put on top (`4. Expenses > 4.3 Submitting a claim`) | clean chunks, headings help matching and citations | needs real headings; very short sections give weak vectors |

**Example of the difference.** Same handbook, same question. A fixed chunk starts mid-word (`d expenses for family members...`) and covers three unrelated sections. A recursive chunk starts cleanly but can begin with the tail of the previous table. A structure chunk is exactly:

```
4. Expenses and Reimbursement > 4.3 Submitting a claim
Submit claims in ExpenseDesk within 30 days of the date of the expense. ...
```

The app uses **recursive, size 800, overlap 100** (see results below for why).

---

## Quick start

You need Python 3.11+, Docker, and [Ollama](https://ollama.com).

```bash
# 1. models
ollama pull nomic-embed-text      # embeddings
ollama pull qwen2.5:7b            # answers

# 2. python deps
python3 -m venv .venv && source .venv/bin/activate
pip install pymupdf "psycopg[binary]" pgvector numpy fastapi uvicorn streamlit

# 3. database (Postgres + pgvector)
docker compose up -d

# 4. put the PDF at data/northwind-handbook.pdf, then load it
python ingest.py                  # chunks + embeds + stores (39 / 41 / 89 chunks)

# 5. try it from the command line
python answer.py "What is the deadline to submit an expense claim?"

# 6. or run the app (two terminals, venv active in both)
uvicorn api:app --reload          # terminal 1: API on :8000
streamlit run app.py              # terminal 2: UI on :8501
```

Call the API directly:

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is the deadline to submit an expense claim?", "k": 5}'
# -> {"answer": "...", "sources": [{"tag": "S1", "page": 7, "score": 0.755, "text": "..."}, ...]}
```

To use your own PDF, replace the file in `data/` and adjust `SKIP_PAGES` in `blocks.py` (cover and contents pages are skipped on purpose).

---

## Project layout

| File | Job |
|---|---|
| `extract_pdf.py` | PDF to text per page (PyMuPDF) |
| `clean_pdf.py` | remove lines that repeat on at least half the pages (headers/footers), with digits masked so "Page 7" and "Page 12" match |
| `blocks.py` | PDF to ordered blocks labelled `h1` / `h2` / `text`, using font size; sorted top-to-bottom, left-to-right |
| `chunk_fixed.py`, `chunk_recursive.py`, `chunk_structure.py` | the three chunkers |
| `embedder.py` | text to vectors via Ollama |
| `db.py` | table, insert, nearest-neighbour search |
| `ingest.py` | runs the whole ingest chain for each chunker |
| `search.py` | CLI: show the top chunks for a question, for each chunker |
| `compare.py` | scores the chunkers on 16 test questions |
| `sweep.py` | scores 10 size/overlap settings on the same questions |
| `answer.py` | retrieve + prompt + LLM answer with citations |
| `api.py` | FastAPI wrapper (`POST /ask`) |
| `app.py` | Streamlit UI |
| `data/` | the test PDF and its answer key |

---

## The test document

`data/northwind-handbook.pdf` is a made-up 20-page company handbook (14 chapters plus appendices, with repeating headers and footers). It was written to contain specific traps, and `data/ANSWER-KEY.md` lists 16 questions with the correct answer and where the evidence lives. Every question exists to expose a known RAG weakness:

| # | Question (short) | What it tests |
|---|---|---|
| 1, 10, 11 | sick leave, remote work abroad, leave carry-forward | near-duplicate sections |
| 2 | "money back" for a late-night cab | vocabulary mismatch (handbook says "reimbursable") |
| 3, 7, 13 | claim deadline, standby allowance, Learning Fridays | a later notice overrides the body text |
| 4 | who approves INR 60,000 | multi-hop (section points to Appendix B) |
| 5 | what is ERR-5503 | exact code lookup |
| 6 | SEV1 acknowledge time | answer lives in a table row |
| 8, 9, 12, 14, 15 | Friday deploys, lost laptop, notice period, PR approvals, gifts | short answers, numbers, conditions |
| 16 | stock option vesting | not in the document (should say "I don't know") |

---

## Results

### What was measured

For each question we retrieve the top chunks and check whether the **evidence text from the answer key appears in them**. This measures retrieval only: "was the right passage handed to the model?" It does not measure whether the model then wrote a correct answer (see Limits).

Question 16 has no answer in the handbook, so **15 is the best possible score**.

### Three chunkers (800 characters, 100 overlap), questions answered within top 1 / 3 / 5

| | fixed | recursive | structure |
|---|---|---|---|
| chunks | 39 | 41 | 89 |
| top 1 | 7 | 9 | 8 |
| top 3 | 11 | 13 | 13 |
| top 5 | 13 | **15** | 14 |

### Size and overlap sweep

| setting | chunks | avg chars | top 1 | top 3 | top 5 |
|---|---|---|---|---|---|
| fixed 300 / 40 | 103 | 298 | 7 | 12 | 13 |
| fixed 800 / 0 | 34 | 785 | 6 | 14 | 15 |
| fixed 800 / 100 | 39 | 782 | 7 | 11 | 13 |
| fixed 1500 / 180 | 21 | 1443 | 8 | 14 | 15 |
| recursive 300 / 40 | 127 | 223 | 8 | 12 | 12 |
| recursive 500 / 60 | 72 | 395 | 8 | 13 | 14 |
| recursive 800 / 0 | 39 | 689 | 8 | 14 | 15 |
| **recursive 800 / 100** | 41 | 685 | **9** | 13 | **15** |
| recursive 800 / 200 | 44 | 693 | **9** | 14 | 14 |
| recursive 1500 / 180 | 20 | 1402 | 6 | 11 | 13 |
| structure (one per section) | 89 | 322 | 8 | 13 | 14 |

### What the numbers say

- **Around 800 characters is the sweet spot.** 300-character chunks lose context (recursive 300: 12 at top 5). 1500-character chunks blur topics (recursive 1500: worst top 1 and top 3).
- **Overlap barely matters here.** 0, 100 and 200 are within one question of each other. This handbook has few answers that straddle a cut.
- **The chunker matters less than the failure cases.** Totals differ by 1 to 2 questions out of 16, which is within noise. The real lesson is *which* questions each one misses.
- **Where they differ** (rank at which the evidence first appeared, lower is better):
  - Fixed was late on Q3 (conflict) and Q6 (table): rank 7. Its chunks mix several topics, so their vectors are blurry (likely cause, not verified chunk by chunk).
  - Structure was late on Q2 (vocabulary mismatch): rank 7. The one-line night-cab rule sits inside a long section about everything claimable (likely cause, not verified).
  - Multi-hop Q4 was found at rank 4 by all three, so **k must be at least 5**.

---

## Answer quality (the full pipeline, 16 questions)

Retrieval is only half the system, so the 16 questions were also run end to end (`python answer.py --all`, `qwen2.5:7b`, temperature 0, top 5 chunks) and the answers were read against `ANSWER-KEY.md` by hand.

| Result | Count | Questions |
|---|---|---|
| Correct | **14** | 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 |
| Partial | 1 | 2 (cab) |
| Wrong | 1 | 4 (approver for INR 60,000) |

What stands out:

- **Notices overriding the body text worked (Q3, Q7, Q13).** The prompt rule "a later notice wins" did its job, and each answer cited the notice.
- **The unanswerable question was handled correctly (Q16).** The reply was "I couldn't find this in the handbook." One question proves little (see Limits), but it is the behaviour we wanted.
- **Q4 is a reasoning failure, not a retrieval failure.** The right chunks were in the prompt. The answer even quotes the approval matrix ("above INR 50,000 is approved by the VP Finance") and then concludes "the stricter approver, the Department head, applies". The correct answer is VP Finance. A 7B local model can contradict itself when combining two sources.
- **Q2 is a partial answer.** The retrieved text was right (the "cab fares for work done after 21:00 IST" rule was in the top 5), but the model described how to file a claim and never said whether the late-night cab is reimbursable.

This is one run, graded by hand, on 16 questions. Treat it as a smoke test, not a benchmark.

---

## Known failure cases

These are real behaviours observed in this project, kept on purpose:

1. **Vocabulary mismatch.** "Money back" versus "reimbursable". Embeddings help but are not perfect; structure chunking ranked this at 7.
2. **Exact codes.** `ERR-5503` is a string, not a meaning. Embedding scores for it were weak (around 0.5 to 0.6). Keyword search (BM25) is the right tool.
3. **Policy versus later notice.** Both chunks must be retrieved, and the prompt must tell the model that notices override the main text. Works in the example at the top.
4. **Multi-hop.** The 4.4 section only says "see Appendix B". The answer needs both chunks, which only works with a large enough k.
5. **Unanswerable questions.** A similarity cutoff cannot tell "not in the document" from "hard to match": junk scored 0.61 to 0.63 while real hits scored 0.59 to 0.67. "I couldn't find this" has to come from the prompt, and should be measured.
6. **Tables arrive as fragments.** PDF extraction returns table cells as separate blocks. They stay in order and mostly stay together, but there is no real table understanding.
7. **Chunks cross section boundaries.** The example at the top shows it: the best chunk starts with the tail of a cab-fares table and then runs into Section 4.3. It worked because the right paragraph was inside.
8. **Orphan headings.** Recursive chunks can end with a heading whose paragraph is in the next chunk.
9. **Right chunks, wrong answer.** In Q4 the correct text was retrieved and the small model still picked the wrong approver. Better retrieval cannot fix this; a stronger model, a step-by-step instruction, or a self-check can.
10. **Contents page skipped on purpose.** It mentions every topic and would match almost any question.

---

## Limits (be honest about what this does not show)

- **Answer quality was graded once, by hand.** 14 of 16 correct, 1 partial, 1 wrong (see above). There is no automatic grader yet and the run was not repeated, so a re-run could differ slightly. Automatic grading is planned for Week 5.
- **16 questions is a small sample.** One question moves a score by 6 percentage points.
- **One synthetic document.** Real PDFs (scans, two columns, complex tables) will break the extraction step first.
- **Scores are not comparable across chunkers** because chunk lengths differ.
- **A 7B local model** follows instructions less reliably than a large hosted one.

---

## What's next

| When | What |
|---|---|
| Week 5 | proper evals: grade answers automatically, track "I don't know" behaviour |
| Week 6 | hybrid search (keyword BM25 plus vectors, merged with reciprocal rank fusion), reranking with a cross-encoder, query rewriting, metadata filters |
| Week 15 | real table parsing |
| Later | contextual chunks, parent-child chunks, an HNSW index when the data grows |
