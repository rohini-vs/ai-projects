# Week 4 Learnings: Docs Q&A (naive RAG)

What I would tell someone starting this project, and what I would change next time.

---

## 1. The biggest lesson: build the test set before the pipeline

I wrote a test document with deliberate traps and a 16-question answer key *before* tuning anything. That turned arguments ("recursive feels better") into numbers ("recursive found 15 of 15 answerable questions in the top 5; fixed found 13"). Without the key I would have looked at one or two good-looking results and picked a chunker by feel.

The key also exposed things a spot check never would: the vocabulary-mismatch question that structure chunking ranked 7th, and the table question that fixed chunking ranked 7th.

**Next time:** write the questions first, for every project.

## 2. Chunking is not one decision

- Size: about 800 characters was best on this document. 300 lost context, 1500 blurred topics.
- Overlap: made almost no difference here. I expected it to matter more. It only helps when answers straddle a cut.
- Method: recursive (cut at paragraph, sentence, word boundaries) beat blind fixed-size cuts, and structure-aware chunks gave clean sections but were weaker on vocabulary mismatch.

Totals differed by only 1 to 2 questions out of 16. That is noise. The useful information was in *which* questions failed, not in the totals. A single hit-rate number hides the failure cases.

## 3. Overlap is counted in characters, and recursive overlap is "up to"

This confused me at first, so it is worth writing down.

- Fixed-size overlap is exactly N characters, even if that repeats half a word.
- Recursive overlap is a *maximum* made of whole sentences. If the last sentence is longer than the overlap number, nothing is repeated. To get real overlap, set it larger than a typical sentence (100 rather than 20).

## 4. Embedding scores are compressed, so they are not a confidence

Unrelated text scored about 0.53 against a query. Junk for an unanswerable question scored 0.61 to 0.63, while genuine hits scored 0.59 to 0.67. There is no threshold that separates "no answer" from "hard to match".

So scores are good for **ranking** and bad for **deciding**. "I don't know" has to come from the prompt (and be evaluated), not from a similarity cutoff.

## 5. Heading text inside the chunk helps

Putting `4. Expenses and Reimbursement > 4.3 Submitting a claim` on top of a chunk gives the embedding extra meaning and gives citations a human name. On the expense-deadline question, structure chunks put both needed chunks at ranks 1 and 2 with a clear score gap before the rest.

The cost: it only works if the document has real headings, and tiny sections give weak vectors.

## 6. Embeddings are weak at exact strings

`ERR-5503` is an identifier, not a meaning. The embedder ranked it acceptably but with low scores. Anything like a code, SKU, or error number wants keyword search (BM25). That is why hybrid search is on the Week 6 list.

## 7. PDF extraction is where real projects start to hurt

The retrieval code was the easy part. The hard part was getting clean text:

- `get_text()` returns text in file order, with headers and footers mixed in and lines hard-wrapped.
- Removing headers and footers by "appears on at least half the pages" worked, but would also delete a real sentence that repeats that often.
- Sorting blocks by their bottom edge scrambled table rows. Sorting by (page, top edge, left edge) fixed it.
- Headings were found by font size, which only works for documents that use size to mark headings.
- Tables arrive as fragments. They stay in order, but there is no table structure.

**Next time:** look at the extracted text of every page before building anything on top of it.

## 8. Conflicts need both chunks and an explicit rule

For "deadline to submit a claim", the body says 30 days and a later notice says 45. Retrieval has to find both, and the prompt has to say "a notice overrides the main text". With both in place the answer came out right and mentioned both numbers. Retrieval alone cannot solve this; the instruction in the prompt is half the fix.

## 9. Multi-hop needs a bigger k

The approval question was found at rank 4 by all three chunkers, so k=3 would have lost it. k=5 is the minimum that worked on these 16 questions. That has a cost: five 800-character chunks per prompt.

## 10. Keep strategies side by side in one table

Adding a `strategy` column and keeping every chunker's output in the same table made the comparison and the size sweep almost free. I could add a new setting without touching the others and re-run the same 16 questions.

## 11. Good retrieval does not guarantee a good answer

Running all 16 questions end to end gave 14 correct, 1 partial and 1 wrong. The wrong one (who approves INR 60,000) had the right chunks in the prompt. The model even quoted the rule that says "above INR 50,000: VP Finance" and then answered "Department head". A 7B local model can fail at the *combining* step even when search did its job.

So the two halves need separate measurements: retrieval (did the right text arrive?) and generation (did the model use it correctly?). When an answer is wrong, the first question is which half failed, because the fixes are different. Retrieval failures want better chunks, hybrid search or reranking. Generation failures want a stronger model, a clearer prompt, or a verification step.

---

## What surprised me

- Overlap mattered much less than I expected.
- The chunk the app showed as the best source began with the tail of an unrelated table, and it still worked. Chunk boundaries that look ugly are not necessarily wrong for retrieval, but they waste prompt space.
- The simplest option (recursive, 800/100) was best or tied for best. The cleverer structure-aware chunker did not win.
- Most of my confusion was about *what the numbers mean* (overlap in characters, what "16" referred to, what the score means), not about the code.

## What I would do differently

1. Write the answer key first and keep it out of the ingested data.
2. Look at the extracted text before chunking anything.
3. Grade answers automatically. I graded the 16 answers by hand once (14 correct, 1 partial, 1 wrong), which is slow, not repeatable, and misses small changes.
4. Add a section heading to recursive chunks too, so citations can say "Section 4.3" and not only "page 7".
5. Test an unanswerable question set larger than one. A single "I don't know" case proves very little.

## Known failure cases (kept, not hidden)

Vocabulary mismatch, exact codes, notices overriding the body text, multi-hop questions, unanswerable questions, table fragments, chunks that cross section boundaries, orphan headings. Details and examples are in the README.

## What this sets up

| Next | Why |
|---|---|
| Week 5, evals | grade answers automatically and measure "I don't know" |
| Week 6, hybrid search + reranking | fixes exact codes and ranking quality |
| Week 15, table parsing | fixes table fragments |
