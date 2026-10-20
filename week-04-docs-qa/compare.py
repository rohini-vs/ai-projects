import re

STRATEGIES = ["fixed", "recursive", "structure"]
K = 10          # how many chunks to retrieve per question

# How "found" is decided, from the answer key's evidence text:
#   any  : one of the phrases in a retrieved chunk is enough
#   all  : every phrase must appear somewhere in the retrieved chunks
#          (the answer needs several facts, e.g. policy + a later notice)
#   same : every phrase must appear in ONE chunk (tests that a table row stays whole)
QA = [
    ("How many days of sick leave do I get, and when do I need a medical certificate?",
     "Near-duplicate sections", "all",
     ["12 days of paid sick leave", "3 or more consecutive working days"]),
    ("How do I get my money back for a cab I took for work late at night?",
     "Vocabulary mismatch", "any",
     ["cab fares for work done after 21:00 IST"]),
    ("What is the deadline to submit an expense claim?",
     "Conflict: policy vs later notice", "all",
     ["within 30 days of the date of the expense", "extended from 30 days to 45 days"]),
    ("Who has to approve an expense claim of INR 60,000?",
     "Multi-hop (see Appendix B)", "all",
     ["listed in Appendix B", "INR 50,001 to INR 200,000"]),
    ("What does ERR-5503 mean and what should I do?",
     "Exact code lookup", "any",
     ["Downstream collector unavailable"]),
    ("How quickly must a SEV1 incident be acknowledged, and how often are updates posted?",
     "Table row kept whole", "same",
     ["Complete outage, data loss or security breach", " 5 minutes"]),
    ("How much is the standby allowance for the primary on-call engineer?",
     "Conflict: policy vs later notice", "all",
     ["standby allowance of INR 3,000 per week", "from INR 3,000 to INR 3,500"]),
    ("Can I deploy to production on a Friday?",
     "Negation / short answer", "any",
     ["There are no Friday deployments"]),
    ("How long do I have to report a lost laptop?",
     "Two numbers in one sentence", "any",
     ["within 2 hours of discovering the loss"]),
    ("How many days can I work from another country?",
     "Near-duplicate sections", "any",
     ["limited to 10 working days in a calendar year"]),
    ("How much unused annual leave can I carry forward?",
     "Near-duplicate sections", "any",
     ["Up to 10 unused days can be carried forward"]),
    ("What is the notice period for an L5 engineer?",
     "Number depends on level", "any",
     ["90 days for L5 and above"]),
    ("When are Learning Fridays?",
     "Conflict: policy vs later notice", "all",
     ["second Friday afternoon of every month", "second and fourth Friday"]),
    ("How many approvals does a pull request touching payments need?",
     "Condition inside a rule", "any",
     ["need 2 approvals"]),
    ("Above what value must I declare a gift from a customer?",
     "Number plus deadline", "any",
     ["worth more than INR 5,000"]),
    ("What is the stock option vesting schedule?",
     "Unanswerable (pointer only)", "any",
     ["individual offer letters"]),
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def rank_of(texts: list[str], rule: str, phrases: list[str]):
    """Rank (1 = best) by which the evidence is found, or None if not found."""
    texts = [norm(t) for t in texts]

    def first(ph):
        return next((i for i, t in enumerate(texts, 1) if ph in t), None)

    if rule == "any":
        found = [r for r in map(first, phrases) if r]
        return min(found) if found else None
    if rule == "all":
        found = [first(p) for p in phrases]
        return None if None in found else max(found)
    # same
    return next((i for i, t in enumerate(texts, 1) if all(p in t for p in phrases)), None)


if __name__ == "__main__":
    from db import connect, search
    from embedder import embed_query

    conn = connect()
    ranks = {s: [] for s in STRATEGIES}

    print(f"{'#':>2}  {'what it tests':<34}{'fixed':>7}{'recursive':>11}{'structure':>11}")
    for n, (q, trap, rule, phrases) in enumerate(QA, 1):
        qvec = embed_query(q)
        row = []
        for s in STRATEGIES:
            texts = [h["text"] for h in search(conn, qvec, s, K)]
            r = rank_of(texts, rule, phrases)
            ranks[s].append(r)
            row.append("-" if r is None else str(r))
        print(f"{n:>2}  {trap:<34}{row[0]:>7}{row[1]:>11}{row[2]:>11}")

    print("\n(number = rank at which the evidence was found; '-' = not in the top "
          f"{K})\n")
    print(f"{'found within top...':<38}{'fixed':>7}{'recursive':>11}{'structure':>11}")
    for k in (1, 3, 5):
        counts = [sum(1 for r in ranks[s] if r is not None and r <= k) for s in STRATEGIES]
        print(f"{'top ' + str(k) + ' (out of ' + str(len(QA)) + ')':<38}"
              f"{counts[0]:>7}{counts[1]:>11}{counts[2]:>11}")