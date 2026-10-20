# Week 4 Learnings: How RAG Works

These notes explain the concepts behind this project, what we found when we tested them, and what we would do differently. You can read them without running anything.

---

## The short version

**The goal:** ask questions in plain English about a PDF and get an answer with page citations, using only local models.

**The approach (RAG):** don't give the model the whole document. First *search* for the few passages that probably contain the answer, then ask the model to answer *using only those passages*.

**The test document:** a made-up 20-page company handbook, written with deliberate traps (a rule that a later notice overrides, a code that must be matched exactly, a question with no answer, and so on) plus an answer key of 16 questions. It lets us measure what works instead of guessing.

**The result in one line:** with a well-chosen chunker the right passage was found for every answerable question, and the full system answered 14 of 16 correctly. The two misses taught more than the 14 hits.

---

# Part 1. The concepts

## 1. The problem RAG solves

A language model only knows what was in its training data. It has never seen your handbook. Ask it "What is the deadline to submit an expense claim?" and it will guess something plausible and generic.

You could paste the whole handbook into every prompt, but long prompts are slow, expensive, and the model gets distracted by irrelevant text.

**RAG (Retrieval-Augmented Generation)** splits the job in two:

1. **Retrieval:** find the few passages most likely to contain the answer.
2. **Generation:** give only those passages to the model and tell it to answer from them.

A useful way to think about it: the model is a smart person taking an open-book exam, and retrieval chooses which pages to open.

**Key consequence:** the answer can only be as good as the passages retrieved. If the right passage is not found, a perfect model still answers wrongly. That is why retrieval and answering are measured separately later in these notes.

## 2. The whole pipeline

```
INGEST (done once per document)

  PDF
   │  extract text, with page numbers and font sizes
   ▼
  clean         remove repeated headers and footers
   │
   ▼
  blocks        paragraphs and headings, in reading order
   │
   ▼
  chunks        cut into searchable pieces
   │
   ▼
  embeddings    each chunk becomes a list of numbers (its meaning)
   │
   ▼
  vector database (Postgres + pgvector)


QUERY (every question)

  question ──► embedding ──► nearest chunks (top 5)
                                  │
                                  ▼
                       prompt: "answer only from these sources"
                                  │
                                  ▼
                         language model ──► answer + citations
```

Two ideas worth noticing:

- **Ingest and query are separate.** Slow work (reading the PDF, computing embeddings) happens once. Answering a question only reads from the database, so it starts instantly.
- **The three query steps are independent.** Search, prompt building and the model call can each be swapped without touching the others. Replacing the database, the model, or the prompt changes one function.

## 3. Getting clean text out of a PDF

This is the least glamorous step and the one that quietly decides everything after it. A PDF is a set of instructions for drawing characters at positions, not a document with paragraphs. Several problems appear immediately:

- **Reading order.** A tool may return text in the order it was stored in the file, not the order a human reads it. Tables are the worst case: sorting blocks by their *bottom* edge scrambles rows, while sorting by (page, top edge, left edge) restores them.
- **Headers and footers.** "Company name | Handbook | Version 4.2" and "Page 7" appear on every page. Left in, they pollute every chunk and add false matches. A reliable way to remove them: find lines that repeat on at least half of the pages, after replacing digits with a placeholder so "Page 7" and "Page 12" count as the same line. Caution: a real sentence that repeats that often would be deleted too.
- **Hard-wrapped lines.** Lines end mid-sentence where the page ends. Join the lines of a block back together.
- **Headings.** A PDF has no "this is a heading" marker. A practical trick is to use font size: the most common size is the body text, and noticeably larger sizes are headings (the biggest is a chapter, the next a section).
- **Pages to skip.** A contents page mentions every topic, so it matches almost every question. Skip it.
- **Tables.** A table arrives as separate fragments, one per cell or row. They stay in order, but there is no real table structure. Proper table parsing is a topic of its own.

**Takeaway:** look at the extracted text of every page before building anything on top of it. Mistakes here are invisible later, because the model just sees whatever text you gave it.

## 4. Chunking

**What it is.** Cutting the document into pieces. A chunk is the unit that is embedded, searched, and pasted into the prompt.

**Why it exists.** One embedding for a whole document would be a blurry average of everything in it. A question about expense deadlines would not stand out against a vector that also contains leave policy, on-call rules and laptop returns.

**The central trade-off:**

| Chunk size | Good | Bad |
|---|---|---|
| Small (about 300 characters) | precise: the vector is about one thing | loses context; an answer can be split across chunks |
| Large (about 1500 characters) | keeps context | blurry vector; wastes prompt space; more distracting text |

Roughly 800 characters worked best on this document. That number is a starting point to test, not a rule.

### Overlap

The next chunk repeats the end of the previous one so an answer that straddles a cut still appears whole in at least one chunk.

A tiny example, size 40 characters, overlap 20, text: `Cats sleep a lot. Dogs bark at night. Birds sing at dawn. Fish swim all day.`

```
Chunk 1: Cats sleep a lot. Dogs bark at night.
Chunk 2: Dogs bark at night. Birds sing at dawn.      <- "Dogs bark at night." repeated
Chunk 3: Birds sing at dawn. Fish swim all day.
```

Two details that cause confusion:

- Overlap is counted in **characters**. In fixed-size chunking it is exactly that many characters and can repeat half a word.
- In sentence-aware chunking overlap is a **maximum**, built from whole sentences. If the last sentence is longer than the overlap number, nothing is repeated at all. So set overlap larger than a typical sentence (100, not 10).

### Three strategies

**1. Fixed-size.** Glue all text together and cut every N characters.
- Simple, but cuts mid-word and mixes unrelated topics in one chunk.

**2. Recursive.** Try to cut at the biggest natural boundary first (blank line, then line break, then sentence end, then space), only going smaller if a piece is still too big. Then glue pieces back together up to the size limit.
- Never cuts mid-sentence. Still ignores sections, so one chunk can end in one section and start another. It can also leave a heading stranded at the end of a chunk, away from its paragraph.

**3. Structure-aware.** One chunk per document section, with the heading path put on top of the text:

```
4. Expenses and Reimbursement > 4.3 Submitting a claim
Submit claims in ExpenseDesk within 30 days of the date of the expense. ...
```

- The heading words become part of what is embedded, which helps matching. They also give citations a human name ("Section 4.3"). A table stays in one chunk.
- It needs real headings, and very short sections give weak vectors. If a section is too long it must be cut with the recursive method.

**Takeaway:** there is no single best chunker. Each one loses in a different place, which is why you test them on questions rather than choose by feel.

## 5. Embeddings

**What they are.** An embedding is a list of numbers (768 of them here) that represents the *meaning* of a text. Texts with similar meaning get similar numbers, even when they share no words.

```
"Can I enroll late?"                 -> numbers close to ->  "You can still join after the start date."
"How do I install Docker?"           -> numbers far from  ->  "You can still join after the start date."
```

That is the whole advantage over keyword search, which would find no overlap between "enroll late" and "join after the start date".

**How they are produced.** A neural network trained so that similar meanings land near each other. It reads the whole sentence, so the same word in different contexts gets different numbers.

**Rules that matter:**

- **Use the same model for documents and questions.** Vectors from different models are not comparable. Changing the model means re-embedding everything.
- **Follow the model's prefix convention.** Some models were trained with a label in front of the text: for example `search_document: ` before stored text and `search_query: ` before questions. Skipping it quietly lowers quality.
- **Embed in batches.** One huge call is slow and shows no progress.

## 6. Similarity and scores

**Cosine similarity** measures the angle between two vectors: 1 means pointing the same way (similar), 0 means unrelated. Databases often report **cosine distance**, which is `1 - similarity`, so smaller means closer.

**The score is not a confidence.** Two things to know:

- Different embedding models use different score ranges. One model may score unrelated text near 0.01, another near 0.5. Scores from different models, or even from differently chunked data, are not comparable.
- A mediocre match can still score high. So a threshold ("reject anything below 0.65") cannot reliably separate "no answer exists" from "hard to match". On this project, real answers and junk fell in overlapping score ranges.

**Takeaway:** use scores to **rank**, not to decide. Detecting "not in the document" has to come from the prompt, and should itself be measured.

## 7. The vector database

**pgvector** adds a `vector` column type to Postgres and operators to find the nearest vectors.

```sql
SELECT content, 1 - (embedding <=> query_vector) AS score
FROM chunks
ORDER BY embedding <=> query_vector     -- <=> is cosine distance, closest first
LIMIT 5;
```

**Exact versus approximate search.**

- Exact search compares the question with every row. It always finds the true best matches and is instant for a few hundred rows.
- Approximate search (an index such as **HNSW**) narrows down to a region of likely matches first. It is much faster at millions of rows, at the cost of occasionally missing the absolute best match.

Start exact. Add an index when the data is large enough that exact search is slow.

**A design tip:** a `strategy` column that labels which chunker produced each row lets several chunk sets live in one table and be compared side by side without touching each other.

## 8. Top-k

**k** is how many chunks to retrieve per question. Returning only one is risky: the answer may be split across several chunks, or the best-ranked one may not be the right one. Returning many wastes prompt space and distracts the model.

A question that needs two facts from different places (for example, "see Appendix B" in one section and the actual numbers in the appendix) may only find the second fact at rank 4. So k=3 would lose it. k=5 was the smallest value that worked here, and the right value should be tested, not guessed.

## 9. Grounded prompting and citations

The prompt has two parts:

- **Instructions** (fixed): how the model must behave.
- **User prompt** (changes every time): the question and the retrieved sources.

The instructions that mattered:

1. **Use only the sources.** Never outside knowledge.
2. **Cite every fact** with a source tag. Label each retrieved chunk `[S1] (page 7)`, `[S2] (page 18)` so the model can point at it and the app can show the page.
3. **A later notice overrides the main text.** Real documents contain amendments. If the body says 30 days and a later notice says 45, both chunks must be retrieved *and* the model must be told which one wins. Retrieval alone cannot solve this.
4. **If the sources do not contain the answer, say so.** A specific refusal sentence ("I couldn't find this in the handbook") that the model repeats exactly gives a result you can test.

**Takeaway:** the prompt is the bridge between search and the model. A weak prompt lets the model ignore the sources and invent an answer.

## 10. Measuring a RAG system

This is the concept that changes how you work.

### Build the test before you tune

Write the test questions and their correct answers **first**, with the evidence text that proves each answer. Then every change (a different chunk size, a different chunker) can be scored on the same questions.

Why this matters: without it, you look at one or two good-looking results and decide a chunker "feels better". Real results disagree with feelings. A fixed test set turns an argument into a number, and it exposes failures a casual check would never find.

### Measure retrieval and answering separately

A wrong answer can come from two different places, and the fixes differ:

| Where it failed | What happened | What fixes it |
|---|---|---|
| **Retrieval** | the right passage never reached the model | better chunks, hybrid search, reranking |
| **Generation** | the right passage was there, and the model still got it wrong | stronger model, clearer prompt, a verification step |

### How retrieval was scored

For each question, check whether the answer-key evidence text appears in the top k chunks ("hit at k"). Three rules cover the cases:

- **any:** one evidence phrase is enough.
- **all:** every phrase must appear somewhere in the retrieved chunks (the answer needs the rule *and* the amendment).
- **same:** every phrase must be in one chunk (tests that a table row stays whole).

Include at least one question that has **no answer** in the document, so you can check the system says "I couldn't find this" instead of guessing.

### Be honest about the limits

- 16 questions is a small sample: one question moves a score by about 6 points. Differences of one or two questions are noise.
- "The evidence is in the top 5" does not mean "the answer is correct".
- Scores from different chunkers are not directly comparable.

---

# Part 2. What we found

These are the results of the project's test. If you rebuild this, expect similar patterns, not identical numbers.

## Chunking results

On the retrieval test (15 answerable questions, top 5):

| Chunker | Found the evidence for |
|---|---|
| fixed, 800 chars / 100 overlap | 13 of 15 |
| recursive, 800 / 100 | **15 of 15** |
| structure-aware | 14 of 15 |

The totals are close, so the patterns matter more:

- **Fixed-size ranked conflicts and tables late.** Its chunks mix several topics, which blurs their meaning.
- **Structure-aware ranked one question late:** a one-line rule (reimbursement for late-night cabs) sat inside a long section about everything claimable, and the question said "money back" while the handbook said "reimbursable". Small chunks are precise, but the wording gap still hurt.
- **Recursive was never the worst.** Its chunks are the size of a paragraph or two and do not cut sentences.

## Size and overlap

| Setting | Result |
|---|---|
| around 300 characters | worse: not enough context per chunk |
| around 800 characters | best overall |
| around 1500 characters | worse at the top ranks: blurry chunks |
| overlap 0, 100, 200 | within one question of each other |

Overlap mattered much less than expected. It helps when answers straddle a cut, and this document had few of those. Do not tune it before size.

## Full pipeline results

Running all 16 questions through retrieval plus the model, graded by hand against the answer key: **14 correct, 1 partial, 1 wrong.**

- **Amendments worked.** Three questions where a later notice overrides the body text were answered with the notice's value, because the prompt said notices win.
- **The unanswerable question was handled.** The model replied that it could not find the answer.
- **Exact codes worked acceptably** but with low similarity scores (embeddings are weak at identifiers).
- **One answer was wrong although the right chunks were present.** The question asked who approves a claim of INR 60,000. The model quoted the rule ("above INR 50,000: VP Finance") and then concluded "Department head". A small local model can fail when it has to combine two sources, even when retrieval did its job.
- **One answer was partial.** The model explained how to file a claim and never said that late-night cab fares are reimbursable, which was the actual question.

## Failure patterns to expect

| Pattern | What happens | Likely fix |
|---|---|---|
| **Different wording** | question says "money back", document says "reimbursable" | hybrid search; query rewriting |
| **Exact codes** | an identifier like `ERR-5503` has no "meaning" to match | keyword search (BM25) alongside vectors |
| **Amendments and conflicts** | both the rule and the notice are needed | retrieve enough chunks; tell the model which wins |
| **Multi-hop** | answer needs two passages (a pointer plus its target) | k of at least 5; later, agentic retrieval |
| **Unanswerable questions** | a score threshold cannot tell "no answer" from "hard to match" | handle it in the prompt, and test it |
| **Tables as fragments** | cells arrive as separate blocks | a real table parser |
| **Chunks crossing sections** | a chunk starts in one section and runs into the next | structure-aware chunking |
| **Orphan headings** | a heading ends one chunk, its paragraph starts the next | structure-aware chunking |
| **Right chunks, wrong answer** | the model combines sources badly | stronger model, step-by-step instructions, a self-check |

---

# Part 3. What we would do differently

1. **Write the answer key first, and keep it out of the ingested data.** If the key were stored in the database, retrieval would find the answers in the key itself and the test would prove nothing.
2. **Read the extracted text of every page before chunking.** Extraction mistakes look like retrieval mistakes later.
3. **Grade answers automatically.** Grading 16 answers by hand once is slow, not repeatable, and misses small changes. Automatic grading comes next.
4. **Give recursive chunks a section heading too.** Then citations can say "Section 4.3, page 7" and not only "page 7", and matching improves.
5. **Test more than one unanswerable question.** One "I couldn't find this" proves very little.
6. **Test the answer step on each chunker, not just retrieval.** Only the recursive chunker was run through the model, so the other two are untested end to end.
7. **Start simpler when it is enough.** Keyword search is faster and cheaper, and works well when users use the document's own words. Add vectors when measurements show keyword search misses things.

---

# Part 4. Key takeaways

1. **RAG = retrieval + generation.** Answers are limited by what retrieval hands over.
2. **Clean extraction is the foundation.** Headers, reading order and headings decide the quality of everything after.
3. **Chunking is a trade-off with no universal winner.** Test size and method on your own questions. About 800 characters worked here, and overlap barely mattered.
4. **Embeddings capture meaning, not exact strings.** Great for paraphrases, weak for codes and identifiers.
5. **Use scores to rank, not to decide.** They are not confidence values.
6. **The prompt is half the system.** Citations, "notices win", and "say you don't know" are instructions, not retrieval features.
7. **Write the test first.** A fixed set of questions with known answers turns opinions into measurements.
8. **Measure retrieval and generation separately.** The fix depends on which half failed.
9. **Each approach fails differently.** That is the reason hybrid search, reranking and better evaluation exist.

## What this sets up

| Next | Why |
|---|---|
| Automatic evaluation (a larger question set, scored faithfulness and "I don't know" behaviour) | replaces hand grading and lets changes be compared quickly |
| Hybrid search (keyword plus vectors), reranking, query rewriting | addresses wording gaps and exact codes |
| Real table parsing | addresses table fragments |
