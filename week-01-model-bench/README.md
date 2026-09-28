# Week 01 — Local Model Bench + Tokenizer From Scratch

Part of an [AI Engineering Journey](../README.md) — a 21-week, project-based path to AI engineering using local models.

**What this project does:**
1. Benchmarks local LLMs (speed, memory, quality) so model choices are based on measurement, not guesswork.
2. Builds a byte-pair encoding (BPE) tokenizer from scratch to show *why* non-English languages cost more tokens with most models.

See [LEARNINGS.md](./LEARNINGS.md) for the full write-up of what each experiment showed and why it matters.

---

## Setup

**Prerequisites**
- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) for Python dependency management

```bash
# Install dependencies
uv sync

# Pull the models used in this benchmark
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:7b-instruct-q8_0
```

**A note on hardware:** local LLM speed and memory numbers are hardware-dependent — chip, memory bandwidth, and available RAM all shape the results. Run the benchmark yourself and record your own machine's specs alongside the output; that's the only way to know what to expect on your setup. The results below were produced on one laptop-class machine and are included as a reference point, not a target to match. Absolute numbers will differ on other hardware, but the relative comparisons (Q4 vs Q8, 3B vs 7B) should hold directionally across most machines.

---

## Project structure

```
week-01-model-bench/
├── src/
│   ├── ollama_client.py   # thin wrapper around the Ollama HTTP API
│   ├── prompts.py         # the 10-prompt test suite (5 categories)
│   ├── bench.py           # speed + memory benchmark
│   ├── score.py           # automatic quality scoring
│   ├── token_cost.py      # English vs Tamil token cost, real models
│   ├── tiny_bpe.py        # BPE tokenizer, implemented from scratch
│   └── bpe_lab.py         # trains 4 tiny tokenizers, compares them
├── scratch/               # early experiments, not part of the pipeline
│   ├── hello_ollama.py    # first Ollama API call
│   └── bpe_steps.py       # BPE worked step by step
├── data/                  # English + Tamil text corpus (gitignored)
├── results/               # CSV output from each run (gitignored)
├── LEARNINGS.md           # write-up of what each experiment showed
└── README.md
```

---

## How to run

**1. Speed and memory benchmark**
```bash
uv run python src/bench.py --models llama3.2:3b,qwen2.5:7b,qwen2.5:7b-instruct-q8_0
```
Runs all 10 prompts through each model, prints a summary table, and saves per-prompt results to `results/bench_<timestamp>.csv`.

**2. Quality scoring**
```bash
uv run python src/score.py
```
Reads the most recent benchmark CSV and automatically grades the factual, reasoning, coding, and JSON answers. Summaries are printed for manual review, since there's no simple rule to check them automatically.

**3. Token cost: English vs Tamil, real models**
```bash
uv run python src/token_cost.py --models llama3.2:3b,qwen2.5:7b
```
Sends 6 matched English/Tamil sentence pairs to each model and compares token counts.

**4. From-scratch BPE tokenizer experiment**
```bash
uv run python src/bpe_lab.py
```
Trains four tiny tokenizers (256 merges each) on different combinations of the English/Tamil corpus in `data/`, then measures tokens-per-character on held-out text.

---

## Results

### Speed and memory

| Model | Memory | Prefill tok/s | Decode tok/s | Total time (10 prompts) |
|---|---|---|---|---|
| llama3.2:3b (Q4) | 2.3 GB | 511 | 50.3 | 5.9s |
| qwen2.5:7b (Q4) | 4.6 GB | 259 | 24.3 | 13.5s |
| qwen2.5:7b-instruct (Q8) | 7.8 GB | 282 | 21.8 | 13.9s |

*Implied memory bandwidth (decode tok/s × model size): ~116 GB/s (3B), ~112 GB/s (7B Q4), ~170 GB/s (7B Q8) — found by multiplying each model's decode speed by its memory footprint. Comparing this against your chip's rated memory bandwidth (from its spec sheet) shows how close a run gets to saturating it: here, the Q8 run came closer to its theoretical peak than either Q4 run did, likely because unpacking 4-bit weights adds compute overhead that partly offsets the smaller read.*

### Quality (8 auto-scored questions across 4 categories; 2 summaries reviewed manually)

| Model | Factual | Reasoning | Coding | JSON | Total | Fenced JSON |
|---|---|---|---|---|---|---|
| llama3.2:3b | 2/2 | 1/2 | 2/2 | 2/2 | 7/8 | 0 |
| qwen2.5:7b (Q4) | 2/2 | 2/2 | 2/2 | 2/2 | **8/8** | 1 |
| qwen2.5:7b-instruct (Q8) | 2/2 | 2/2 | 2/2 | 1/2 | 7/8 | 1 |

**Failures:**
- `llama3.2:3b` on reason-1 ("60 km in 45 min, speed in km/h"): answered `7.33`, correct answer is `80`.
- `qwen2.5:7b-instruct-q8_0` on json-2: returned `"price": "1499"` (string) instead of `1499` (number).

**Note on sample size:** 6 of 8 scored questions were answered correctly by all three models. Only 2 questions actually separated them. This is too small a test set to draw strong conclusions — treat these as directional, not definitive. A larger eval set (week 5 of the main plan) would sharpen this.

### Token cost: English vs Tamil, real models

*Pending — run `src/token_cost.py` and record results here.*

| Model | English tokens (6 sentences) | Tamil tokens (6 sentences) | Tamil costs |
|---|---|---|---|
| llama3.2:3b | — | — | — |
| qwen2.5:7b | — | — | — |

### From-scratch BPE: tokens per character on held-out text (lower is better)

| Tokenizer trained on | English | Tamil |
|---|---|---|
| No merges (raw bytes) | 1.00 | 2.76 |
| English-only | 0.50 | 2.74 |
| Tamil-only | 1.00 | 0.48 |
| English + Tamil (mixed) | 0.56 | 0.58 |

Corpus: 6,891 characters of original English text and 4,707 characters (12,917 bytes) of original Tamil text, written for this project — see [LEARNINGS.md](./LEARNINGS.md) for why original text was used instead of a scraped source. 256 merges per tokenizer.

---

## Key findings

1. **Q8 didn't beat Q4 on quality** in this test, despite costing 3.2 GB more RAM and running ~10% slower. Q4 is the sensible default; only reach for Q8 if you find a task where Q4 measurably underperforms.
2. **The 3B model is a legitimate choice for lightweight tasks.** It runs 2× faster than the 7B models in a third of the memory, and only lost on a multi-step arithmetic question — not on extraction, coding, or JSON.
3. **Both Qwen models occasionally wrapped JSON output in markdown fences** despite explicit instructions not to. Any code parsing structured output from these models needs to strip fences defensively — the theme of week 2.
4. **Tamil starts at ~2.76 bytes per character vs. 1.00 for English**, before any tokenizer training happens. This head start gap is the root cause of every downstream cost difference.
5. **A shared vocabulary hurts the smaller-resource language more.** Under a mixed tokenizer, English lost ~12% efficiency relative to its own dedicated tokenizer, but Tamil lost ~21%. This is a small-scale replica of why real multilingual LLMs — trained on vocabularies where English dominates the data — are measurably less token-efficient on languages like Tamil.

---

## What's next

Week 2 builds a structured data extractor and starts handling exactly the kind of format inconsistency (fenced JSON, string-vs-number) that showed up in this week's quality scoring.
