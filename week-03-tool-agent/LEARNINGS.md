# Tool-Calling Agents: Concepts and Findings

This is the write-up behind [week 3](./README-week03.md) of a 21-week, project-based path to AI engineering using local models.

## Intent

The goal was to build a tool-calling loop by hand: a model decides whether it needs a function, a real function executes, and the genuine result feeds back so the model can continue or give a final answer — the same mechanism every agent framework automates later, built here without one.

---

## Learning 1: An easy task never exercises tool-calling at all

Given `847 × 23`, both models solved it from memory without ever engaging the tool-use instruction — correctly, in this case (`19,481`). A test built on a task the model can already do reliably proves nothing about whether tool-calling actually works. Only a genuinely hard task — `6847 × 3928` — forced a real decision.

## Learning 2: Model size affected self-awareness of limits, not just correctness

On the hard multiplication (`6847 × 3928`, correct answer `26,895,016`):

| Model | Behavior | Answer |
|---|---|---|
| llama3.2:3b | Answered directly, confidently | `27,011,036` — **incorrect** |
| qwen2.5:7b | Recognized the task exceeded reliable mental math, called `calculator` | `26,895,016` — correct |

The smaller model didn't fail by refusing to answer — it failed by answering wrong with the same confidence as a correct answer would carry. The larger model's advantage here wasn't more arithmetic skill; it was correctly identifying when to stop trusting itself.

## Learning 3: A fabricated tool-use transcript can be indistinguishable from a real one

In one generation, a model produced two separate JSON tool calls and a final prose answer, all in a single response — none of it executed by the actual tools. A simplified version of what it produced:

```
{"tool": "word_counter", "args": {"text": "..."}}
{"tool": "calculator", "args": {"expression": "word_counter(text) * 12"}}
There are 9 words in the sentence, and that number multiplied by 12 equals 108.
```

The first parsing approach tried to interpret the *entire* response as one JSON value, which failed outright (multiple JSON objects back to back isn't valid JSON), causing the whole blob — fake tool calls included — to be treated as a plain final answer. No tool ever ran. The final numbers (`9`, `108`) were correct only because the model did the reasoning in its head while pretending to use tools — a coincidence, not a guarantee.

**The fix:** parse only the *first* JSON object in a response, using `json.JSONDecoder().raw_decode()`, and discard everything generated after it:

```python
decoder = json.JSONDecoder()
data, _ = decoder.raw_decode(text)  # parses one object, ignores trailing text
```

This enforces the system prompt's own rule ("respond with ONLY this JSON") in code rather than relying on the model to honor it — the same principle as week 2's schema validation, applied to a sequence of decisions instead of a single structured answer.

## Learning 4: A model used a tool selectively, not for everything available

Given a two-part question — count the words in a sentence, then multiply that count by 12 — the model called `word_counter` for real (confirmed by execution logs) but computed the final multiplication (`9 × 12`) itself, without a second tool call. It used the tool for the part it couldn't do reliably by inspection and skipped it for arithmetic simple enough to trust.

## Learning 5: Executing model-chosen code requires restricting what it can reach, not just checking its shape

A tool's argument can be perfectly well-typed (a string, as expected) and still be dangerous to execute blindly. `calculator`'s expression is run through `eval()` with empty builtins and names dictionaries, which permits arithmetic operators but blocks access to any function — `__import__`, `open`, or anything else that could reach outside the expression itself.

---

## Results

| Task | llama3.2:3b | qwen2.5:7b |
|---|---|---|
| 847 × 23 (easy) | Direct answer, correct | Direct answer, no tool used |
| 6847 × 3928 (hard) | Direct answer, **incorrect** | Tool call, correct |
| Multi-step (word count → ×12) | Not tested | 1 real tool call (`word_counter`), final step computed directly, correct |

---

## Summary

- A tool-calling test is only meaningful if the task is hard enough that the model plausibly can't do it unaided — an easy task proves nothing.
- A model confidently answering incorrectly, rather than deferring to a tool, is a real failure mode distinct from simply "getting it wrong" — it carries no signal that anything is off.
- A model can generate a complete, plausible-looking fake tool-use transcript with no real execution behind it; a correct final answer is not proof that a tool was actually used.
- Enforcing "the response must be only this JSON" in code — parsing just the first valid object and discarding the rest — removes dependence on the model actually following that instruction.
- A capable model will use a tool selectively, reserving it for the part of a task it can't reliably do itself rather than routing everything through it.
- Restricting what a dynamically-executed expression can access (no builtins, no names) is a minimum safeguard whenever code chosen by a model gets run.

**Code for this project:** see the [README](./README-week03.md) for setup and how to reproduce every result here.
