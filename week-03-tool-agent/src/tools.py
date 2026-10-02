def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression and returns the result as a string."""
    try:
        # restricted eval: no builtins, no names — just arithmetic
        print(f"Evaluating expression: {expression}")
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def word_counter(text: str) -> str:
    """Counts the number of words in a piece of text."""
    return str(len(text.split()))


TOOLS = {
    "calculator": calculator,
    "word_counter": word_counter,
}