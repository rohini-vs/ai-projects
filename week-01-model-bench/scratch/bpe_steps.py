text = "abcabcabd"
ids = list(text.encode("utf-8"))
print("Input bytes:", ids)

print(len("hello"), len("hello".encode("utf-8")))
tamil = "நான்"
print(len(tamil), len(tamil.encode("utf-8")))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(ids)
print(stats)

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(ids)
print(stats)


def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)   # glue the pair into one symbol
            i += 2               # skip both original pieces
        else:
            out.append(ids[i])   # leave this piece alone
            i += 1
    return out

best = max(stats, key=stats.get)   # the most frequent pair
print(best)

ids = merge(ids, best, 256)
print(ids, len(ids))

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}                          # remembers every glue we did, in order
    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges

merges = train("abcabcabd", 258)
print(merges)

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

new_ids = encode("abcabd", merges)
print(new_ids)

def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}   # every raw byte, on its own
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]        # glue the bytes, not the numbers
    return vocab

vocab = build_vocab(merges)
print(vocab[256])
print(vocab[257])

def decode(ids, merges):
    vocab = build_vocab(merges)
    all_bytes = b"".join(vocab[i] for i in ids)
    return all_bytes.decode("utf-8")

text_back = decode([257, 256, 100], merges)
print(text_back)