# Local Model Bench: Concepts and Findings

This is the write-up behind [week 1](./README.md) of a 21-week, project-based path to AI engineering using local models — a from-scratch look at how local LLMs actually perform, and why tokenization quietly disadvantages some languages. No frameworks were used; every number here comes from the Ollama HTTP API and plain Python, measured directly rather than sourced from elsewhere.

No prior experience with inference or tokenization is assumed.

---

## Part 1: How fast is "fast"? Prefill vs. decode

Every time a prompt is sent to a local model, two different things happen:

1. **Prefill** — the model reads the entire prompt at once, processing it in parallel. This is fast.
2. **Decode** — the model writes its answer one token at a time, and each new token depends on everything written so far. This is slow, because it can't be parallelized the same way.

This was measured directly across three models:

| Model | Prefill (reading) | Decode (writing) |
|---|---|---|
| llama3.2:3b | 511 tokens/sec | 50.3 tokens/sec |
| qwen2.5:7b (4-bit) | 259 tokens/sec | 24.3 tokens/sec |
| qwen2.5:7b (8-bit) | 282 tokens/sec | 21.8 tokens/sec |

Reading is roughly **10× faster** than writing, across every model. This is why a long prompt with a short answer feels snappy, but a short prompt asking for a long answer feels slow — the bottleneck is almost always decode, not prefill. Shortening a prompt barely helps latency; shortening the expected answer helps far more.

## Part 2: Why decode speed depends on model size, not just "how powerful" a model is

Decode speed is mostly determined by **how much data has to move through memory**, not by how much "thinking" the model does. During decode, the model reads essentially all of its weights to generate each single token. So:

> **max decode speed ≈ (how fast the machine moves data) ÷ (how big the model is)**

On the machine used here, memory moves at roughly 200 GB/s according to its spec sheet — check any given machine's own specs for its rated figure. Multiplying each model's actual decode speed by its size in memory gives:

| Model | Memory used | Decode speed | Implied bandwidth used |
|---|---|---|---|
| llama3.2:3b | 2.3 GB | 50.3 tok/s | ~116 GB/s |
| qwen2.5:7b (4-bit) | 4.6 GB | 24.3 tok/s | ~112 GB/s |
| qwen2.5:7b (8-bit) | 7.8 GB | 21.8 tok/s | ~170 GB/s |

The 8-bit model came closest to using the full rated memory bandwidth. Both 4-bit models used noticeably less of it. A plausible explanation: 4-bit weights have to be unpacked (converted back to a usable precision) before use, which costs some compute time that partly cancels out the benefit of moving less data. This isn't confirmed by this experiment alone, but it's consistent with the numbers, and it's a good example of a benchmark raising a question it doesn't fully answer.

**Practical takeaway:** compressing a model (quantization) shrinks its memory footprint predictably, but the speed gain isn't as simple as "half the bits, twice the speed." Measuring beats assuming.

## Part 3: Does more compression actually cost quality? Testing Q4 vs Q8

The standard expectation is that smaller quantization (like 4-bit) is faster and smaller but loses some accuracy compared to a less-compressed version (like 8-bit). The same model, Qwen 2.5 7B, was run at both 4-bit and 8-bit precision through 8 automatically-gradable questions (facts, reasoning, code, structured JSON output):

| Model | Score |
|---|---|
| qwen2.5:7b (4-bit) | **8 / 8** |
| qwen2.5:7b (8-bit) | 7 / 8 |

The 8-bit version didn't win. Its one miss wasn't a reasoning failure — it returned a price as the text `"1499"` instead of the number `1499` in a JSON response, a formatting slip rather than a factual error.

**Caveat:** 8 questions is a tiny sample. One question is worth 12.5 percentage points, so this doesn't prove Q8 is "worse." What it does show is that on this test, going from 4-bit to 8-bit — which cost 3.2 GB more memory and ran about 10% slower — bought no measurable quality improvement. That's a useful data point for choosing a default: if the more expensive option isn't clearly winning, default to the cheaper one, and only upgrade when a specific task shows it matters.

## Part 4: Grading answers automatically, and where that breaks down

Eyeballing 30 model responses and deciding "looks right" by feel doesn't scale and isn't very rigorous. Simple automatic checks were built instead:

- **Factual/reasoning questions:** does the expected keyword appear in the answer?
- **JSON output:** does it parse, and does it match the expected object exactly — including type. This is how the `"1499"` vs `1499` issue was caught; skimming the output alone would likely have missed the quotes.
- **Code:** run the model's function against real test cases (e.g. `assert fib(10) == 55`) in an actual Python subprocess. If it doesn't crash and the asserts pass, it's correct — the same idea behind coding benchmarks like HumanEval, since running code is a much stronger check than reading it and guessing.

This approach is fast, free, and consistent, but it only knows what it was told to check for. If a model answered "eighty" instead of "80," the keyword check would have called it wrong despite being correct. Rule-based grading works well for anything with one right answer; it falls apart on open-ended tasks like summaries, which were reviewed manually instead.

The proper fix for open-ended grading is called **LLM-as-judge** — using another model to grade the answer against a rubric — which is a large enough topic to be its own project later on. The lesson here is smaller but still important: automatic doesn't mean infallible. Knowing exactly what a scoring script does and doesn't check is a prerequisite for trusting its numbers.

## Part 5: Why Tamil costs more tokens than English — built from first principles

### What a tokenizer actually does

Before a model reads text, a tokenizer chops it into pieces called tokens. Models don't work with raw text — they work with a fixed vocabulary of these pieces, and that vocabulary is learned from whatever text the tokenizer was trained on, using an algorithm called **byte-pair encoding (BPE)**.

The idea:
1. Start by treating text as raw bytes (every character in any language reduces to 1 or more bytes).
2. Find whichever two neighboring bytes/pieces appear together most often.
3. Merge that pair into one new symbol.
4. Repeat, thousands of times.

Frequently-seen patterns get squashed into a single token; rare patterns stay as several small pieces. The entire algorithm was implemented from scratch — no libraries — to see exactly how it works:

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
```
`get_stats` counts how often each pair of neighboring symbols shows up. `merge` glues the winning pair into a new symbol. Training a tokenizer is calling these two functions in a loop, a few hundred or thousand times, remembering every merge along the way. That saved list of merges — nothing more — *is* the tokenizer.

### A worked example, step by step

Take the text `"abcabcabd"` (9 characters). Each letter becomes a byte value: a=97, b=98, c=99, d=100.

**Starting point:**
```
text:  a   b   c   a   b   c   a   b   d
ids:  97  98  99  97  98  99  97  98 100
```

**Round 1 — count every neighboring pair:**

| Pair | Count |
|---|---|
| (97, 98) — "ab" | 3 |
| (98, 99) — "bc" | 2 |
| (99, 97) — "ca" | 2 |
| (98, 100) — "bd" | 1 |

`(97, 98)` wins with 3 occurrences. It's merged into a new symbol, `256` (meaning "ab"):

```
before: 97  98  99  97  98  99  97  98 100   (9 symbols)
after:      256    99   256    99   256   100   (6 symbols)
```

**Round 2 — count the pairs again, on the new 6-symbol sequence:**

| Pair | Count |
|---|---|
| (256, 99) — "ab"+"c" | 2 |
| (99, 256) — "c"+"ab" | 2 |
| (256, 100) — "ab"+"d" | 1 |

`(256, 99)` wins. It's merged into symbol `257` (meaning "abc"):

```
before: 256   99  256   99  256  100   (6 symbols)
after:       257       257     256  100    (4 symbols)
```

Training stops here (only 2 merges in this toy example). The learned tokenizer is nothing more than this small recipe:

```
merges = {(97, 98): 256, (256, 99): 257}
```

**Applying it to new, unseen text — `"abcabd"`:**

```
raw bytes:            97  98  99  97  98 100
apply (97,98)->256:       256    99  256 100
apply (256,99)->257:          257       256 100
```

Result: `[257, 256, 100]` — 3 tokens instead of 6 raw bytes, using merges learned from a *different* sentence. Decoding this list (reversing the same recipe) reconstructs the text exactly: `"abcabd"`. This encode/decode round trip is lossless by construction — whatever gets compressed, decoding fully restores, which is essential, since a tokenizer that lost information would corrupt every piece of text it touched.

This tiny example is the entire mechanism behind every modern LLM tokenizer, just scaled up from 2 merges to 100,000+.

### The head start problem

Before any training happens, there's already a large structural gap. English characters are 1 byte each in UTF-8. Tamil characters are typically 3 bytes each. So a sentence with the same meaning in both languages starts out roughly **2.7× longer in raw bytes** for Tamil than for English — before a tokenizer has learned anything.

This was confirmed directly: on held-out text with zero merges applied, English measured 1.00 tokens per character (as expected — one byte per character) and Tamil measured 2.76.

### The experiment: what happens when languages compete for vocabulary

Four tiny tokenizers (256 merges each) were trained on different slices of a matched English/Tamil corpus, then tested on text none of them had seen:

| Tokenizer trained on | English (tokens/char) | Tamil (tokens/char) |
|---|---|---|
| No merges (raw bytes) | 1.00 | 2.76 |
| English text only | 0.50 | 2.74 |
| Tamil text only | 1.00 | 0.48 |
| Both languages mixed | 0.56 | 0.58 |

- **A tokenizer only gets good at what it was trained on.** The English-only tokenizer compressed English by half (1.00 → 0.50) but did almost nothing for Tamil (2.76 → 2.74) — it had never seen the patterns.
- **When trained on both, neither language gets its full share.** English under the mixed tokenizer (0.56) is worse than English under its own dedicated tokenizer (0.50) — about **12% more tokens**. Tamil under the mixed tokenizer (0.58) is worse than Tamil's dedicated result (0.48) — about **21% more tokens**.
- **Tamil loses more from sharing than English does.** With only 256 total merge "slots" to hand out, and English making up more of the combined training text, English's patterns won more of the competition for those slots.

### Why this isn't just a toy result

This is a small, 256-merge version of something that happens at full scale in every real multilingual model. Public training data used to build these vocabularies is overwhelmingly English. When a 100,000+ slot vocabulary is built from that mix, English dominates the competition for merges and ends up highly compressed. Tamil, Hindi, and most other lower-resource languages keep more of their text as expensive, barely-merged byte fragments.

This is why, in practice, the same sentence in Tamil costs noticeably more tokens than in English with most mainstream models — not because the model "understands" Tamil less, but because its tokenizer never had enough Tamil training data to earn Tamil the same number of efficient shortcuts English got. More tokens means slower generation and higher cost for identical meaning, purely as a side effect of how the vocabulary was built.

---

## Summary

- Prefill (reading) is roughly 10× faster than decode (writing); expected-answer length affects latency more than prompt length.
- Decode speed is mostly memory-bandwidth-bound, but not purely so — quantization changes the compute/memory balance in ways worth measuring rather than assuming.
- On an 8-question test, 8-bit quantization showed no measurable quality advantage over 4-bit, despite using more memory and running slower.
- Rule-based quality scoring is fast and reliable for checkable answers (facts, JSON, code) but blind to anything it wasn't explicitly told to check.
- Byte-level BPE tokenization explains, mechanically, why Tamil (and similarly under-represented languages) costs more tokens than English with most models: a smaller share of training data means a smaller share of the learned vocabulary, and therefore fewer efficient shortcuts.
- Numbers generated directly, on real hardware, with real text, are more trustworthy than numbers assumed from general reasoning about how models "should" behave.

**Code for this project:** see the [README](./README.md) for setup and how to reproduce every result here.
