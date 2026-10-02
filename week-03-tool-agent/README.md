# Week 03 — Tool-Calling Agent

Part of an [AI Engineering Journey](../README.md) — a 21-week, project-based path to AI engineering using local models.

## Intent

Weeks 1 and 2 taught a model to produce output matching a fixed contract — a JSON schema. This week asks for something with more agency: the model must decide *whether* it needs help, choose a real function to call, and use that function's actual result to continue reasoning. The goal was to build this loop by hand — no agent framework — since every framework covered later in this plan automates exactly the mechanism built here.

See [LEARNINGS-week03.md](./LEARNINGS-week03.md) for the findings and results.

---

## Setup

Reuses the Ollama setup from previous weeks.

```bash
uv sync
```

No new models this week — `llama3.2:3b` and `qwen2.5:7b` from week 2.

---

## Project structure

```
week-03-tool-agent/
├── src/
│   ├── ollama_client.py  # carried over from week 2, extended with a `stop` sequence option
│   ├── tools.py           # the real functions the agent can call
│   ├── agent.py            # the tool-call contract, parser, and the loop itself
│   └── test_loop.py        # runs single-tool and multi-step questions through the loop
```

`test_prompt.py`, an earlier single-shot (non-looping) test script, was retired once `agent.py` gained a full loop that strictly supersedes it.

---

## The tools

```python
def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression and returns the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def word_counter(text: str) -> str:
    """Counts the number of words in a piece of text."""
    return str(len(text.split()))
```

`calculator` uses `eval()` with empty builtins and names dictionaries. Plain `eval()` executes arbitrary Python, which is unsafe when the input string originates from a model rather than a trusted human — passing empty dictionaries strips access to every built-in function (no `__import__`, no `open`, nothing beyond basic arithmetic operators), so ordinary expressions still work while anything attempting to reach outside the expression fails with a `NameError`.

---

## The contract

```
You can answer directly, or use a tool if you need to.

Available tools:
- calculator(expression: str) — evaluates a basic arithmetic expression, e.g. "23 * 17"
- word_counter(text: str) — counts the number of words in a piece of text

To use a tool, respond with ONLY this JSON, nothing else:
{"tool": "<tool_name>", "args": {...}}

You may use a tool more than once if needed — for example, count words first, then
calculate something using that count. Once you have everything you need, give your
final answer in plain text, not JSON.

If you don't need any tool at all, just answer directly in plain text.
```

The loop in `agent.py` parses only the **first** JSON object found at the start of a response (via `json.JSONDecoder().raw_decode`), executes that tool for real, appends the genuine result back into the transcript, and repeats — up to a fixed step cap — until the model produces a non-JSON, final answer.

---

## How to run

```bash
uv run python src/test_loop.py --model qwen2.5:7b
uv run python src/test_loop.py --model llama3.2:3b
```

---

## Results

### Does task difficulty determine whether a tool gets used?

| Task | llama3.2:3b | qwen2.5:7b |
|---|---|---|
| 847 × 23 (easy) | Answered directly: `19,481` (correct) | Answered directly, no tool used |
| 6847 × 3928 (hard) | Answered directly: `27,011,036` — **incorrect** (actual: `26,895,016`) | Called `calculator` → `26,895,016` (correct) |

### Multi-step chain (qwen2.5:7b)

Question: *"How many words are in the sentence 'the quick brown fox jumps over the lazy dog', and what is that number multiplied by 12?"*

| Step | Action | Result |
|---|---|---|
| 1 | Model called `word_counter` on the sentence | Real execution → `9` |
| 2 | Model computed `9 × 12` itself, no second tool call | `108` (correct) |

---

## Known issues

- A `stop=["}\n"]` sequence was added to `generate()` intended to cut generation short right after a single tool-call JSON object. It did not reliably take effect during testing — the model frequently continued generating past that point regardless. Correctness does not depend on it: parsing only the first JSON object (`raw_decode`) makes the system correct independent of whether the stop sequence fires, so this was left as an open efficiency gap rather than pursued further.

---

## Key findings

1. **An easy task doesn't exercise tool-calling at all.** Both models solved `847 × 23` from memory/pattern, without ever engaging the tool-use instruction. A test has to use a task the model is actually likely to get wrong unaided, or it proves nothing.
2. **Model size affected whether the model recognized its own limits**, not just raw correctness. On the hard multiplication, `llama3.2:3b` answered confidently and incorrectly; `qwen2.5:7b` correctly identified the task as beyond reliable mental math and delegated to the tool.
3. **A model can fabricate an entire fake tool-use transcript in one generation** — multiple JSON tool calls and a final answer, all invented, with zero real execution — and still land on the mathematically correct answer by coincidence. This is indistinguishable from a real success unless tool execution is explicitly verified. See [LEARNINGS-week03.md](./LEARNINGS-week03.md) for the concrete example and the fix.
4. **The model used the tool selectively, not universally** — it delegated word-counting (something it can't do reliably by inspection) but computed a trivial multiplication (9 × 12) itself once it had the count, rather than issuing a second tool call for arithmetic it could safely handle.

---

## What's next

Week 4 builds a documents Q&A system (naive RAG) — chunking, embeddings, retrieval, and answering with citations, the foundation the next several weeks build on.
