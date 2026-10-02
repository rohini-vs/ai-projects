# Structured Extraction: Concepts and Findings

This is the write-up behind [week 2](./README-week02.md) of a 21-week, project-based path to AI engineering using local models.

## Intent

The goal was to turn "usually works" into "works reliably enough for a pipeline": define an exact schema for a correct answer, validate every model response against it, and when a response doesn't comply, feed the specific error back to the model and retry rather than discarding the result.

---

## Learning 1: Schema validation checks structure, not meaning

A Pydantic schema guarantees the right *shape* of data — correct fields, correct types. It says nothing about whether the content is actually right.

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

On a clean test posting, every field validated successfully — including `location`, which came back as `"Chennai office"` instead of `"Chennai"`. A valid string, imprecise content, and no validation error to catch it. Closing this gap needs either a stricter schema (an enum of known cities, a regex) or a separate semantic check — a real limitation of structural validation alone.

## Learning 2: Passing validation isn't proof a model understood the input

A second, deliberately ambiguous posting included the line *"open to remote for the right candidate but prefer Bangalore-based folks."* Run without any retry logic:

| Model | `remote` returned | Validation |
|---|---|---|
| qwen2.5:7b | `"open to remote for the right candidate"` (the original phrase, preserved as text) | ✗ Failed — not a boolean |
| llama3.2:3b | `true` | ✓ Passed |

The model that passed didn't reason more carefully — it guessed. The model that failed preserved a genuine ambiguity the schema had no way to represent. A confident wrong guess and a careful, honest answer can be structurally indistinguishable to a validator.

## Learning 3: Specific error feedback can turn a failure into a success

The retry loop catches a validation error and tells the model exactly what was wrong, rather than just asking again generically:

```
1 validation error for JobPosting
remote
  Input should be a valid boolean, unable to interpret input
```

Fed this error and the chance to retry, `qwen2.5:7b` corrected `remote` into a valid boolean on its second attempt. It wouldn't resolve the ambiguity unprompted, but could when explicitly told the field required a boolean. Naming the exact problem, not just flagging failure, is what made the retry effective.

## Learning 4: A reasoning model's "thinking" cost needs to be measured, not assumed

One model tested this week (Qwen3, 4B) spends tokens on an internal reasoning phase before producing a final answer. On this task — simple field extraction, not a task that needs multi-step reasoning — that overhead didn't stay fixed. Measuring the actual thinking length at increasing token budgets showed it scaling almost exactly with whatever budget it was given, on both an easy and a hard posting alike, rather than converging on an answer:

| Token budget | Thinking length used | Answer produced? |
|---|---|---|
| 300 | 300 (maxed out) | No |
| 1,200 | ~1,200 (maxed out) | No |
| 3,000 | ~3,000 (maxed out) | No |
| 4,500 | ~4,485 (maxed out) | No |

This model was excluded from the rest of the week's comparison: for this task, in this setup, its reasoning overhead had no reliable upper bound. The broader lesson carries beyond this one model — a reasoning model's token cost for a given task is worth measuring directly before relying on it, rather than assuming a bigger budget will eventually be enough.

---

## Results

Retry-loop comparison across the two models that completed reliably:

| Model | Posting | Result | Attempts |
|---|---|---|---|
| llama3.2:3b | easy | ✓ | 1 |
| llama3.2:3b | hard | ✓ | 1 |
| qwen2.5:7b | easy | ✓ | 1 |
| qwen2.5:7b | hard | ✓ | 2 |

---

## Summary

- A schema validates structure, not meaning — a field can be the right type and still hold the wrong content.
- Passing validation is not proof of correct understanding; a confident guess and a careful answer can look identical to a validator.
- Feeding a model the specific validation error, not a generic "try again," is what makes a retry effective.
- A reasoning model's token overhead should be measured against the actual task, not assumed to shrink with patience or a bigger budget.

**Code for this project:** see the [README](./README-week02.md) for setup and how to reproduce every result here.
