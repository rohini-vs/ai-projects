# Week 02 — Structured Extractor

Part of an [AI Engineering Journey](../README.md) — a 21-week, project-based path to AI engineering using local models.

## Intent

Week 1 surfaced a problem without solving it: local models occasionally wrapped JSON output in markdown fences or returned the wrong type for a field (a price as text instead of a number), despite explicit instructions not to. In a chat window that's a minor annoyance. In a pipeline that calls `json.loads()` on the output, it's an unhandled exception.

This project's goal was to build a **structured extractor**: a function that takes messy, free-text input and reliably returns data matching an exact schema, by validating every response and automatically retrying — with the specific validation error fed back to the model — when it doesn't comply.

See [LEARNINGS-week02.md](./LEARNINGS-week02.md) for the full write-up, including a worked example of the retry mechanism and a detailed account of a model that was tested and excluded from this week's comparison.

---

## Setup

Reuses the Ollama setup from week 1.

```bash
uv sync
```

**Models used this week:**
```bash
# Already pulled in week 1:
# ollama pull llama3.2:3b
# ollama pull qwen2.5:7b

# New this week:
ollama pull qwen3:4b
```

`qwen3:4b` was pulled and tested, then excluded from the final comparison — see [Results](#results) and [LEARNINGS-week02.md](./LEARNINGS-week02.md) for why.

---

## Project structure

```
week-02-structured-extractor/
├── src/
│   ├── ollama_client.py    # Ollama API wrapper, carried over from week 1 and extended
│   ├── schema.py           # the Pydantic contract a valid extraction must satisfy
│   ├── naive_extract.py    # the "just ask nicely" version, kept to document failures
│   ├── extract.py          # the retry loop: validate, feed back the specific error, retry
│   └── compare_extract.py  # runs extract() across models and test postings
```

---

## How to run

**1. See the naive approach fail** (no retry logic, no fence-stripping)
```bash
uv run python src/naive_extract.py --model qwen2.5:7b
```
Edit `JOB_TEXT` inside the file to switch between the easy and hard test postings.

**2. Run the retry-loop comparison**
```bash
uv run python src/compare_extract.py --models llama3.2:3b,qwen2.5:7b
```
Runs both models against both test postings and reports how many attempts each one needed.

---

## Test data

Two job postings were used, chosen to represent different levels of difficulty:

**"easy"** — clean and unambiguous:
> New job alert - Backend Engineer needed at Flexon Systems, Chennai office. 3-5 yrs exp required. Pay range 12-18 LPA depending on experience. Remote not available, hybrid 3 days/week.

**"hard"** — deliberately ambiguous, with a currency symbol next to numbers and an unlabeled extra qualifier:
> We're hiring! Senior Data Analyst role, open to remote for the right candidate but prefer Bangalore-based folks. Looking for someone with 4+ years, ideally up to 7. Comp is around ₹15-22 LPA, negotiable for the right fit. Company: Verlix Analytics.

The schema both postings are checked against:
```python
class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    min_experience_years: int
    max_experience_years: int
    salary_min_lpa: float
    salary_max_lpa: float
    remote: bool
```

---

## Results

### Naive extraction (no retry, no fence-stripping)

| Model | Posting | Result | What happened |
|---|---|---|---|
| qwen2.5:7b | easy | ✓ Passed | `location` returned as `"Chennai office"` rather than `"Chennai"` — a valid string, imprecise content, not caught by validation |
| qwen2.5:7b | hard | ✗ Failed | `remote` returned as the descriptive text `"open to remote for the right candidate"` instead of a boolean |
| llama3.2:3b | hard | ✓ Passed | Committed to `remote: true` on the same ambiguous text |
| qwen3:4b | hard | ✗ Failed | Returned an empty response — ran out of its token budget while "thinking," never reached an answer |

### With the retry loop

| Model | Posting | Result | Attempts |
|---|---|---|---|
| llama3.2:3b | easy | ✓ | 1 |
| llama3.2:3b | hard | ✓ | 1 |
| qwen2.5:7b | easy | ✓ | 1 |
| qwen2.5:7b | hard | ✓ | **2** |

The one result that matters most: `qwen2.5:7b` on the hard posting went from a hard failure to a success once the specific validation error was fed back to it. See [LEARNINGS-week02.md](./LEARNINGS-week02.md) for the full mechanism and what this does and doesn't prove.

**Known gap:** `compare_extract.py` records only the final successful attempt, not the intermediate ones. What `remote` actually resolved to on `qwen2.5:7b`'s second attempt was not logged or verified. Worth adding attempt-by-attempt logging before drawing firmer conclusions from this result.

### qwen3:4b — tested and excluded

Four attempts to get `qwen3:4b` to disable its "thinking" phase or complete within a reasonable budget all failed:

| Attempt | Approach | Result |
|---|---|---|
| 1 | `think=False` API flag | No effect |
| 2 | Inline `/no_think` soft-switch | No effect |
| 3 | Manually-written empty `<think></think>` tags, raw mode | No effect |
| 4 | Raising the token budget (300 → 1200 → 3000 → 4500) | Thinking length scaled with the budget every time, never converging |

Full detail, including why each method likely failed, is in [LEARNINGS-week02.md](./LEARNINGS-week02.md).

---

## Key findings

1. **Schema validation catches structure, not meaning.** `"Chennai office"` passed as a valid `location` string. A schema can guarantee the right shape of data; it says nothing about whether the content is actually correct.
2. **Passing validation isn't proof a model understood the input.** `llama3.2:3b` passed the ambiguous posting by committing to a guess (`remote: true`); `qwen2.5:7b` initially failed the same posting by preserving the ambiguity as text rather than guessing. The one that failed arguably handled the ambiguity more honestly.
3. **Specific error feedback can convert a failure into a success.** `qwen2.5:7b` needed a validation error telling it exactly what was wrong (`remote` must be a boolean) before it would commit to one — it wouldn't on its own, but could when explicitly required to.
4. **A workaround that's documented for one tool doesn't necessarily work in another.** Three separately-documented ways to disable Qwen3's thinking mode all failed under Ollama specifically. Testing a claim beats trusting the first source that states it, even when that source looks authoritative.
5. **Raising a token budget doesn't fix an unbounded process.** `qwen3:4b`'s thinking length scaled almost exactly with whatever budget it was given, at every level tested. That's a sign of an open-ended process, not one that simply needed more room.

---

## What's next

Week 3 builds a tool-calling agent from scratch — the same "model must return something your code can act on" problem this week solved for JSON extraction, applied to deciding which function to call and with what arguments.
