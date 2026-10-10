# Week 3 Study Material: Attention Mechanisms and GPT Architecture

## Purpose

This standalone study guide covers the attention concepts studied during Week 3. It explains how self-attention gathers context, how Query, Key, and Value work together, why attention scores are normalized, how causal masking works, what multiple attention heads do, and where attention fits into a GPT-style Transformer.

Keep project-specific experiments, implementation details, benchmarks, and results in `LEARNINGS.md`. This file is for conceptual study material.

## 1. Self-attention: gathering context

Consider the sentence:

> The animal didn't cross the road because it was tired.

To interpret `it`, context from other tokens—such as `animal`—may be useful. Self-attention lets each token calculate how much attention to give to other tokens in the same sequence.

The result is that a token can gather information from other tokens rather than being represented only in isolation.

Illustrative example:
- When processing `it`, a model may assign a relatively high attention weight to `animal`.
- The actual attention weights depend on the model, layer, attention head, and input.
- Attention weights do not necessarily correspond directly to human notions of meaning or explanation.

### Connection to earlier concepts

The tokenizer converts text into token IDs. An embedding layer maps token IDs to learned vectors. Attention operates on learned representations of tokens, not by treating token IDs themselves as meaningful similarity scores.

## 2. Query, Key, and Value (Q, K, V)

A useful analogy is a library:

- **Query (Q):** what you are looking for.
- **Key (K):** searchable information used to assess whether something is relevant.
- **Value (V):** the information that can be retrieved and used.

These are mental models for learned numerical representations, not literal questions or labels stored in the text.

For a token embedding \(x\), the model creates Q, K, and V through learned transformations:

\[
Q=xW_Q,\qquad K=xW_K,\qquad V=xW_V
\]

Here, \(W_Q\), \(W_K\), and \(W_V\) are learned weight matrices.

Conceptually:
1. Start with input embedding vectors.
2. Apply separate learned transformations to obtain Query, Key, and Value vectors.
3. Compare Queries with Keys to calculate attention scores.
4. Use the resulting weights to combine Value vectors.

**Remember:** Query and Key determine the attention weights; Value supplies the information that is combined.

## 3. Attention scores and softmax

### Step 1: Calculate scores

A Query is compared with Keys to calculate compatibility scores. For example, while processing `drank` in `The cat drank milk`, an illustrative set of scores might be:

```text
The:    0.5
cat:    2.0
drank:  1.0
milk:   3.0
```

These are invented values for explanation, not outputs from a real model. Attention scores are raw scores, not probabilities.

### Step 2: Scale the scores

Scaled dot-product attention divides the Query–Key dot products by the square root of the key dimension, \(d_k\). This helps keep score magnitudes manageable:

\[
\frac{QK^\top}{\sqrt{d_k}}
\]

### Step 3: Apply softmax

Softmax converts a row of attention scores into normalized attention weights. The weights sum to 1:

\[
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

For the illustrative scores `[0.5, 2.0, 1.0, 3.0]`, the approximate weights are:

| Token | Approximate attention weight |
|---|---:|
| The | 5.2% |
| cat | 23.2% |
| drank | 8.5% |
| milk | 63.1% |

### Step 4: Combine the Value vectors

The model uses the attention weights to combine the corresponding Value vectors. A token whose Value receives a larger weight contributes more strongly to the resulting representation.

### Attention softmax versus output softmax

Softmax can appear in two different places:

- **Inside attention:** normalizes Query–Key scores into attention weights used to combine Value vectors.
- **At the model output:** can convert next-token logits into probabilities used by a decoding strategy.

The mathematical operation is the same, but the inputs and purpose are different.

## 4. Causal masking

GPT-style autoregressive models predict tokens using preceding context. They must not let a position access future tokens when making a next-token prediction.

For the sequence:

```text
The | cat | drank | milk
```

the causal attention permissions are triangular:

| Token being processed | Tokens it may attend to |
|---|---|
| The | The |
| cat | The, cat |
| drank | The, cat, drank |
| milk | The, cat, drank, milk |

A causal mask blocks future positions. It does not block the current token or earlier tokens.

### How masking is applied

Before softmax, blocked attention scores are assigned a very large negative value, commonly represented as negative infinity. Softmax then assigns those positions zero attention weight.

For example, when processing `cat`, a simplified score row might look like:

```text
Before masking: [0.5, 2.0, 1.0, 3.0]
After masking:  [0.5, 2.0, -∞, -∞]
```

The example illustrates the permission pattern, not a universal implementation detail for every model.

### Padding mask versus causal mask

- **Padding mask:** identifies padding positions rather than real input tokens.
- **Causal mask:** prevents a token from attending to future positions.

These masks solve different problems. A model may use both, depending on its architecture and task.

## 5. Multi-head attention

A single attention head uses one set of learned Query, Key, and Value projections. Multi-head attention uses several heads in parallel, each with its own learned projections.

Different heads can learn different patterns of relationships in the same sequence. For example, some heads may learn patterns related to pronouns, nearby words, or grammatical relationships. These are possible patterns, not fixed roles: the model learns its attention behavior during training, and heads are not guaranteed to have a clear human-interpretable function.

For head \(h\), the projections can be written as:

\[
Q_h=XW_Q^{(h)},\quad
K_h=XW_K^{(h)},\quad
V_h=XW_V^{(h)}
\]

Each head computes attention:

\[
\operatorname{Head}_h=
\operatorname{softmax}\left(
\frac{Q_hK_h^\top}{\sqrt{d_k}}
\right)V_h
\]

The model then concatenates the head outputs and applies a learned output projection.

Conceptual flow:
1. Start with the same input representations.
2. Each head creates its own Q, K, and V.
3. Each head calculates attention independently.
4. Combine the outputs from the heads.
5. Pass the combined representation onward.

More heads do not automatically make a model better. Architecture, training, and computational constraints matter too.

## 6. How attention fits into GPT

Consider the prompt:

> The cat sat on the

A GPT-style model generates a continuation autoregressively, generally selecting one token at a time.

High-level flow:

```text
Input text
   ↓
Tokenizer: text → token IDs
   ↓
Token embeddings + positional information
   ↓
Repeated Transformer blocks
   ↓
Final output layer produces next-token logits
   ↓
Decoding strategy selects a token
   ↓
Append the selected token and continue generation
```

### Inside a typical Transformer block

A simplified view of a GPT-style block is:

```text
Input representations
   ↓
Masked multi-head self-attention
   ↓
Residual connection and normalization
   ↓
Feed-forward network (MLP)
   ↓
Residual connection and normalization
   ↓
Output representations for the next block
```

The exact ordering and normalization arrangement vary by architecture. The key point is that attention is not the entire Transformer block: feed-forward computation, residual connections, and normalization are also important.

A GPT model stacks multiple Transformer blocks. The representations are progressively refined as they pass through the network.

### Next-token prediction

At the end of the network, an output layer produces logits for possible next tokens. A decoding strategy—such as greedy selection or sampling—chooses a token. The selected token is appended to the context, and generation repeats.

Causal masking is applied inside self-attention so a position cannot use future tokens that should not be available for its prediction.

## 7. Common misconceptions

1. **“Attention weights are a perfect explanation of a model's reasoning.”**  
   Not necessarily. Attention weights show how the attention computation distributes weight, but they do not automatically provide a complete explanation of model behavior.

2. **“Query, Key, and Value are three names for token IDs.”**  
   No. They are learned vector representations created from input representations by separate transformations.

3. **“Attention scores are already probabilities.”**  
   No. They are raw compatibility scores. Softmax converts them into normalized attention weights.

4. **“Causal masking and padding are the same.”**  
   No. Causal masking blocks future positions; padding masks identify padding positions.

5. **“Every attention head has a fixed, known job.”**  
   No. Heads may learn different patterns, but their functions are learned and not guaranteed to map neatly to human labels.

6. **“Attention is the whole Transformer block.”**  
   No. Typical blocks also contain feed-forward computation, residual connections, and normalization.

7. **“GPT generates the entire answer in one step.”**  
   Standard autoregressive generation proceeds token by token, repeatedly selecting a next token from the model's outputs.

## 8. Quick self-check

Try answering these without looking back:

1. What does self-attention allow a token to do?
2. What are the roles of Query, Key, and Value?
3. How are Q, K, and V created from input representations?
4. What do attention scores represent, and why is softmax applied?
5. What does the scaling factor \(\sqrt{d_k}\) do conceptually?
6. How does causal masking prevent access to future tokens?
7. How does causal masking differ from a padding mask?
8. What is the purpose of multiple attention heads?
9. What other components are present in a typical Transformer block besides attention?
10. How does a GPT-style model use logits to generate text?

## Week 3 takeaway

Self-attention lets tokens gather context. Query–Key comparisons determine attention weights, and those weights control how Value vectors are combined. Multi-head attention performs several learned attention operations in parallel. In GPT-style models, causal masking prevents future-token access, and stacked Transformer blocks process the input before the output layer produces next-token logits for autoregressive generation.
