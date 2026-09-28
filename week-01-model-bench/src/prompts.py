PROMPTS = [
    # factual
    {"id": "fact-1", "category": "factual",
     "prompt": "What is the capital of Australia? Answer in one word.", "expected": "canberra"},
    {"id": "fact-2", "category": "factual",
     "prompt": "Who wrote the novel 'Pride and Prejudice'? Answer with just the name.", "expected": "austen"},

    # reasoning
    {"id": "reason-1", "category": "reasoning",
     "prompt": "A train travels 60 km in 45 minutes. What is its average speed in km/h? Give just the number.",
     "expected": "80"},
    {"id": "reason-2", "category": "reasoning",
     "prompt": "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.",
     "expected": "yes"},

    # coding
    {"id": "code-1", "category": "coding",
     "prompt": "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Return only code."},
    {"id": "code-2", "category": "coding",
     "prompt": "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Return only code."},

    # summarization (longer prompts, so prefill speed shows up)
    {"id": "sum-1", "category": "summarization",
     "prompt": "Summarize in one sentence:\n\nQuantization reduces the precision of a neural network's weights, for example from 16-bit floating point to 4-bit integers. This shrinks the model's memory footprint and speeds up inference because less data must be read from memory for each generated token, at the cost of a small loss in accuracy."},
    {"id": "sum-2", "category": "summarization",
     "prompt": "Summarize in one sentence:\n\nThe city council voted 7-2 on Tuesday to approve a new bike lane network spanning 40 km, funded by a state grant. Construction begins in March and is expected to take eighteen months. Opponents argued that the lanes would remove parking from busy shopping streets."},

    # json output
    {"id": "json-1", "category": "json",
     "prompt": "Extract name, age, city from: 'Priya is 29 and lives in Chennai.' Respond with only a JSON object with keys name, age, city."},
    {"id": "json-2", "category": "json",
     "prompt": "Extract product, price, currency from: 'The ThinkPad X1 costs 1499 USD.' Respond with only a JSON object with keys product, price, currency."},
]