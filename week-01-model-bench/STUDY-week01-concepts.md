# Study Material — Week 1 Concepts
## From raw text to the Transformer: pre-tokenization, token embeddings, and positional embeddings

This guide teaches three connected concepts:

1. **Regex pre-tokenization** — how raw text is split into manageable chunks before BPE tokenization.
2. **Token embeddings** — how token IDs are turned into vectors a neural network can process.
3. **Positional embeddings** — how the model receives information about where tokens occur in a sequence.

The goal is to understand the flow and the mechanics, not just memorize definitions.

---

## 1. The full flow at a glance

A simplified text-to-Transformer pipeline looks like this:

```text
Raw text
   |
   v
Regex pre-tokenization
   |
   v
Text chunks
   |
   v
BPE merges / tokenization
   |
   v
Token IDs
   |
   v
Token embedding lookup
   |
   v
Add positional information
   |
   v
First Transformer layer
```

Each stage has a different job:

- **Pre-tokenization** establishes boundaries between text chunks.
- **BPE tokenization** converts those chunks into vocabulary tokens according to learned merge rules.
- **Token IDs** are integer labels identifying the resulting vocabulary tokens.
- **Token embeddings** map those labels to dense numeric vectors.
- **Positional embeddings** add information about each token's location in the sequence.

Pre-tokenization does **not** itself produce the final token IDs, and token IDs are not the vectors used directly by the Transformer.

---

# 2. Regex pre-tokenization

## 2.1 Why split text before BPE?

Byte Pair Encoding (BPE) builds tokens by applying merge rules to smaller units. If merges could operate freely across every character boundary in the entire input, they could potentially combine material across spaces or punctuation in ways the tokenizer is not intended to allow.

A tokenizer can therefore use a **pre-tokenization** step: first split the text into chunks using rules, then apply BPE within those chunks. The pre-tokenizer establishes boundaries; BPE determines which subword pieces become vocabulary tokens inside those boundaries.

Think of it as two separate decisions:

1. **Where may tokenization work?** Pre-tokenization answers this.
2. **Which pieces should be merged?** BPE answers this.

A pre-tokenization chunk is not necessarily one final token. A chunk may become one token or several tokens after BPE.

## 2.2 A GPT-2-style regular expression

A commonly discussed GPT-2-style pattern is:

```python
pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This pattern is designed to identify different kinds of text chunks. It uses Unicode character properties such as `\p{L}` and `\p{N}`, which are supported by the third-party Python `regex` package. Python's built-in `re` module does not support those `\p{...}` properties in the same way.

Here are the major pieces:

| Pattern fragment | Meaning |
|---|---|
| `'(?:[sdmt]|ll|ve|re)` | Matches selected apostrophe contractions, such as `'s`, `'d`, `'m`, `'t`, `'ll`, `'ve`, and `'re`. |
| ` ?` | Matches zero or one ordinary space. It allows a leading space to stay attached to the following chunk. |
| `\p{L}+` | Matches one or more Unicode letters. |
| `\p{N}+` | Matches one or more Unicode numeric characters. |
| `[^\s\p{L}\p{N}]+` | Matches one or more characters that are not whitespace, letters, or numbers—for example, punctuation and other symbols. |
| `\s+(?!\S)` | Matches a run of whitespace when it is not immediately followed by a non-whitespace character. In effect, this can capture whitespace at the end of a sequence. |
| `\s+` | Matches one or more whitespace characters. |

A few details matter:

- `+` means **one or more** repetitions.
- `?` means **optional** when it follows a character or group.
- `(?:...)` groups alternatives without creating a capturing group.
- `|` means **or**.
- `\s` means whitespace.
- In `[^\s\p{L}\p{N}]`, the `^` at the beginning of the character class means “anything except” the listed categories.
- The order of alternatives matters because the regex engine tries alternatives according to the pattern and its matching rules.

The exact tokenizer implementation and library can affect details, so treat this as a GPT-2-style example rather than a universal regex for every tokenizer.

## 2.3 Walkthrough: a sentence

Consider this input:

```text
Hello world! Isn't BPE great? I've learned 2 things.
```

Using the pattern above in a compatible tokenizer implementation, the pre-tokenization chunks are:

```python
[
    "Hello",
    " world",
    "!",
    " Isn",
    "'t",
    " BPE",
    " great",
    "?",
    " I",
    "'ve",
    " learned",
    " 2",
    " things",
    "."
]
```

Notice the leading spaces on chunks such as `" world"` and `" BPE"`. A space does not always have to be an independent chunk; the pattern can attach a single leading space to the following letters or numbers.

Notice also that `"Isn't"` is split into `" Isn"` and `"'t"` in this example. The contraction alternative can match the apostrophe suffix separately.

These are **pre-tokenization chunks**, not necessarily the final BPE tokens. BPE can split a chunk further or represent it using vocabulary pieces according to its merge rules.

## 2.4 What pre-tokenization does—and does not do

Pre-tokenization does:

- establish boundaries before BPE;
- separate many letter, number, punctuation, contraction, and whitespace patterns;
- influence which character sequences can be considered together during later tokenization.

Pre-tokenization does not:

- assign the final vocabulary ID to every chunk;
- create embedding vectors;
- understand the meaning of the sentence;
- guarantee that each chunk becomes exactly one token.

### Check your understanding

1. Why not let BPE merge across the entire input without any pre-tokenization boundaries?
2. What does `\p{L}+` match?
3. Why might `" world"` retain its leading space?
4. Is `"Hello"` guaranteed to be one final token after BPE? Why or why not?

---

# 3. Token embeddings

## 3.1 A token ID is a label, not a meaning

After tokenization, each token is represented by an integer ID from the tokenizer's vocabulary. For example:

```text
"Hello"  ->  15496
"world"  ->   995
```

These numbers are illustrative; actual IDs depend on the tokenizer and vocabulary.

An ID is an **identifier**. It is not a measurement of meaning. Token ID `995` is not more meaningful, larger, or more important than token ID `15496`. The numbers primarily let software refer to vocabulary entries.

A neural network needs numerical representations that it can process and adjust during training. The model therefore uses an **embedding matrix** to map each token ID to a dense vector.

## 3.2 The embedding matrix

Imagine a vocabulary containing \(V\) tokens and an embedding dimension of \(d\). The token embedding matrix has shape:

\[
E \in \mathbb{R}^{V \times d}
\]

- \(V\): number of tokens in the vocabulary, so there is one row per token ID.
- \(d\): number of numbers in each token's embedding vector.
- Row \(i\) of \(E\): the vector associated with token ID \(i\).

For a toy vocabulary of 10 tokens and an embedding dimension of 3, the matrix has 10 rows and 3 columns:

```text
                 embedding dimensions
                d1      d2      d3
token ID 0     [ ...     ...     ... ]
token ID 1     [ ...     ...     ... ]
token ID 2     [ ...     ...     ... ]
...
token ID 9     [ ...     ...     ... ]
```

In a real language model, the vocabulary may contain tens of thousands of tokens and the embedding dimension may be hundreds or more. The small matrix above is only a teaching example.

## 3.3 Looking up vectors in code

Here is a small NumPy demonstration:

```python
import numpy as np

vocab_size = 10
embedding_dim = 3

# Random values are used only to demonstrate the lookup.
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

token_ids = [3, 7, 3]
vectors = embedding_matrix[token_ids]

print(vectors.shape)  # (3, 3)
```

What happens step by step?

1. `embedding_matrix` contains one row for each of the 10 possible token IDs.
2. The input sequence is `[3, 7, 3]`.
3. Looking up ID `3` returns row 3.
4. Looking up ID `7` returns row 7.
5. The final ID `3` returns row 3 again.
6. The result contains three vectors, each with three numbers, so its shape is `(3, 3)`.

The same ID retrieves the same row when the embedding matrix has not changed. In this example, the values are random and have no learned linguistic meaning.

## 3.4 How do embeddings become useful?

At initialization, embedding vectors may be random or initialized using another scheme. During training, the model adjusts its parameters—including the embedding matrix—so the representations help the model reduce its prediction error.

As a result, embedding vectors can come to reflect useful patterns. Tokens that occur in similar contexts may develop related representations. The model's internal representations can capture syntactic, semantic, and other statistical relationships.

An often-used illustration is:

\[
\text{vector}(\text{king}) -
\text{vector}(\text{man}) +
\text{vector}(\text{woman})
\approx \text{vector}(\text{queen})
\]

This is an illustrative example of relationships that have sometimes appeared in learned embedding spaces, not a rule guaranteed to hold for every model or every embedding.

## 3.5 What token embeddings provide

Token embeddings give the model a learned, dense representation of **which token** is present. But token embeddings alone do not tell the model where that token appears in the sequence. That is the next concept.

### Check your understanding

1. Why is token ID `500` not inherently more meaningful than token ID `20`?
2. If the embedding dimension is 768, how many values are in each token vector?
3. If the input IDs are `[3, 7, 3]`, why do the first and third lookups match?
4. Why do random embedding values not yet represent useful language knowledge?

---

# 4. Positional embeddings

## 4.1 Why does the model need position information?

A sentence is not just a bag of words. Word order affects meaning:

- “The dog bites the man.”
- “The man bites the dog.”

Both sentences contain the same words, but the roles are different because their order differs.

A token embedding represents the token itself. If a sequence representation supplied only token embeddings and no positional information, the model would lack an explicit signal telling it where each token occurs. A Transformer therefore needs a way to represent position or order.

One common approach is to use **positional embeddings**: a vector associated with each position in the sequence.

## 4.2 Adding token and position vectors

In an additive positional-embedding setup, the model adds the token embedding and the positional embedding element by element:

\[
\mathbf{x}_i =
\mathbf{e}_{\text{token}_i} +
\mathbf{p}_i
\]

where:

- \(\mathbf{e}_{\text{token}_i}\) is the token embedding at position \(i\);
- \(\mathbf{p}_i\) is the positional embedding for position \(i\);
- \(\mathbf{x}_i\) is the combined vector passed into the first Transformer layer.

The two vectors must have the same dimension so their values can be added element by element.

## 4.3 A numeric example

Suppose the token embedding for the word `"cat"` is:

\[
\mathbf{e}_{\text{cat}} = [0.2,\ 0.5,\ -0.1]
\]

At position 0, suppose the positional embedding is:

\[
\mathbf{p}_0 = [0.1,\ 0.0,\ 0.2]
\]

Add corresponding elements:

```text
Token embedding:       [ 0.2, 0.5, -0.1]
Position embedding:    [ 0.1, 0.0,  0.2]
                       ------------------
Combined vector:       [ 0.3, 0.5,  0.1]
```

So:

\[
[0.2,\ 0.5,\ -0.1] + [0.1,\ 0.0,\ 0.2]
= [0.3,\ 0.5,\ 0.1]
\]

Now suppose the same token `"cat"` appears at position 2, with a different positional embedding:

\[
\mathbf{p}_2 = [-0.2,\ 0.3,\ 0.4]
\]

Then:

```text
Token embedding:       [ 0.2, 0.5, -0.1]
Position embedding:    [-0.2, 0.3,  0.4]
                       ------------------
Combined vector:       [ 0.0, 0.8,  0.3]
```

The token embedding is unchanged, but the combined representation differs because the position differs.

These numbers are made-up values for arithmetic practice; they are not weights copied from a trained model.

## 4.4 What positional information does—and does not do

Positional information:

- gives the model a signal about token order or location;
- changes the input representation based on where a token occurs;
- works alongside token information rather than replacing it.

It does not mean that the model simply reads a position number as the meaning of a word. The positional vector is another learned or constructed representation that the model processes with the token representation.

**Important qualification:** additive positional embeddings are one approach, not the only approach. Transformer architectures can represent position in other ways, including methods that incorporate positional information into attention. The addition formula above describes the specific additive approach being studied here.

### Check your understanding

1. Why can “dog bites man” and “man bites dog” not be treated as equivalent sequences?
2. What happens when the same token occurs at two different positions?
3. Why must the token and positional vectors have matching dimensions for element-wise addition?
4. Does every Transformer architecture use this exact additive method?

---

# 5. Connect the three concepts

Consider a short input:

```text
Cats sleep.
```

The simplified process is:

1. **Regex pre-tokenization:** split the raw text into chunks based on character patterns. The chunks might preserve spaces with the following word.
2. **BPE tokenization:** use the tokenizer's vocabulary and merge rules to turn each chunk into one or more tokens.
3. **Token IDs:** represent each resulting vocabulary token with its integer ID.
4. **Token embedding lookup:** use each ID to retrieve a vector from the embedding matrix.
5. **Add positional information:** combine each token vector with the vector for its sequence position, if the architecture uses additive positional embeddings.
6. **First Transformer layer:** process the resulting sequence of vectors.

```text
"Cats sleep."
      |
      v
Regex pre-tokenization
      |
      v
Chunks
      |
      v
BPE tokenization
      |
      v
Token IDs: [id₁, id₂, ...]
      |
      v
Embedding lookup
      |
      v
Token vectors: [e₁, e₂, ...]
      |
      v
Add position vectors: eᵢ + pᵢ
      |
      v
Vectors enter Transformer
```

The important distinction is that these stages solve different problems:

| Stage | Main question answered |
|---|---|
| Regex pre-tokenization | Which text chunks are considered separately? |
| BPE tokenization | Which vocabulary tokens represent those chunks? |
| Token IDs | Which vocabulary entries are present? |
| Token embeddings | What dense learned vector represents each token? |
| Positional embeddings | Where in the sequence does each token occur? |

---

# 6. Review questions

Try answering these without looking back.

1. Explain the difference between pre-tokenization and BPE tokenization.
2. What do `\p{L}+` and `\p{N}+` mean in the example regex?
3. Why might a pre-tokenizer attach a leading space to a word chunk?
4. What is the shape of an embedding matrix with vocabulary size \(V\) and embedding dimension \(d\)?
5. What does an embedding lookup do when given token ID `7`?
6. Why does a token ID not carry its own semantic meaning?
7. What changes when the same token appears at a different sequence position?
8. Calculate the sum of `[0.4, -0.1, 0.2]` and `[0.1, 0.3, -0.2]`.
9. Why does a Transformer need positional information in addition to token identity?
10. Describe the flow from raw text to the vectors entering the first Transformer layer.

## Answer key

1. Pre-tokenization splits text into chunks according to rules; BPE applies tokenization/merge rules within those chunks.
2. `\p{L}+` matches one or more Unicode letters; `\p{N}+` matches one or more Unicode numeric characters.
3. The regex's optional leading-space component allows a space to stay attached to the following chunk.
4. \(V \times d\).
5. It retrieves row 7 of the embedding matrix.
6. The ID is an arbitrary vocabulary label; the learned vector associated with that ID carries the representation used by the model.
7. The positional vector changes, so the combined token-plus-position vector changes.
8. `[0.5, 0.2, 0.0]`.
9. Token identity alone does not explicitly indicate the token's place in the sequence, and word order can change meaning.
10. Raw text → regex pre-tokenization → BPE tokenization → token IDs → token embedding lookup → add positional information (in an additive setup) → first Transformer layer.

---

# 7. Final takeaways

- **Pre-tokenization sets boundaries.** It is a text-splitting stage before BPE; its chunks are not necessarily final tokens.
- **BPE produces vocabulary tokens.** The tokenizer maps text chunks to tokens and then to integer IDs.
- **Token IDs are labels.** Their numeric values are not semantic measurements.
- **Embeddings turn IDs into vectors.** Each ID retrieves a row from the embedding matrix.
- **Training shapes the representations.** Randomly initialized vectors do not automatically have meaningful language structure.
- **Position adds order information.** In an additive setup, the model combines token and positional vectors before the first Transformer layer.
- **The concepts form one pipeline.** Text must be tokenized before IDs can be looked up, and the resulting vectors need positional information or another positional mechanism for the model to reason about sequence order.
