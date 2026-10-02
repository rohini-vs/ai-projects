import json

from ollama_client import generate
from tools import TOOLS

SYSTEM_PROMPT = """You can answer directly, or use a tool if you need to.

Available tools:
- calculator(expression: str) — evaluates a basic arithmetic expression, e.g. "23 * 17"
- word_counter(text: str) — counts the number of words in a piece of text

To use a tool, respond with ONLY this JSON, nothing else:
{"tool": "<tool_name>", "args": {...}}

You may use a tool more than once if needed — for example, count words first, then
calculate something using that count. Once you have everything you need, give your
final answer in plain text, not JSON.

If you don't need any tool at all, just answer directly in plain text."""

MAX_STEPS = 4


def try_parse_tool_call(text: str) -> dict | None:
    """Parses only the FIRST JSON object in the response. The system prompt
    requires a tool call to be the entire response — anything the model
    generates after that is drift, not a legitimate continuation, and must
    be discarded rather than trusted."""
    text = text.strip()
    if not text.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    if "tool" in data and "args" in data:
        return data
    return None


def solve(question: str, model: str, verbose: bool = False) -> str:
    transcript = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

    for step in range(1, MAX_STEPS + 1):
        result = generate(model, transcript, temperature=0.0, max_tokens=200, stop=["\nTool result:"])
        response = result["response"].strip()

        if verbose:
            print(f"  raw response: {response!r}")
            weird = [(i, c, hex(ord(c))) for i, c in enumerate(response) if ord(c) > 127]
            if weird:
                print(f"  non-ASCII characters detected: {weird}")

        call = try_parse_tool_call(response)
        if call is None:
            return response  # model gave a final answer, not a tool call

        tool_name = call["tool"]
        tool_fn = TOOLS.get(tool_name)
        if tool_fn is None:
            return f"Error: model requested unknown tool '{tool_name}'"

        tool_result = tool_fn(**call["args"])
        if verbose:
            print(f"  step {step}: called {tool_name}({call['args']}) -> {tool_result}")

        # Feed the model's own tool call AND the real result back into the
        # transcript, so its next generation sees what actually happened.
        transcript += f"\n{json.dumps(call)}\nTool result: {tool_result}\n"

    return f"Error: did not reach a final answer within {MAX_STEPS} steps"