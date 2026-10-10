# Week 2 Study Material: Hugging Face Transformers, Tokenization, Logits, Softmax, Padding, and Attention Masks

## Purpose

This document is a standalone study guide for the concepts covered during Week 2. It explains how text moves through a pretrained classification model, from raw text to a prediction.

This file is conceptual study material. Keep project-specific experiments, benchmarks, implementation results, and observations in `LEARNINGS.md`.

## 1. The big picture

For a text-classification example, the high-level flow is:

1. Start with raw text.
2. Tokenize the text and convert tokens into numerical IDs.
3. Pad sequences when necessary to make examples in a batch the same length.
4. Provide the model with the inputs and relevant masks.
5. Run the model to obtain logits.
6. Apply task-appropriate post-processing. For a typical two-class classification example, softmax can convert logits into normalized probabilities.
7. Return a prediction, such as a sentiment label and score.

Not every task uses exactly the same post-processing. In particular, text generation uses model outputs differently from sequence classification.

## 2. Hugging Face Transformers: library, tokenizer, and model

These are three different things:

- **Transformers library:** A Python library that provides tools for loading and using pretrained models.
- **Tokenizer:** Converts text into tokens and numerical IDs that the selected model expects.
- **Model:** The trained neural network that processes the numerical inputs and produces outputs.

Using the library does not automatically mean that a neural network is being run. The specific code determines what happens.

### Tokenizer-only example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

text = "I love this course!"

tokens = tokenizer.tokenize(text)
input_ids = tokenizer(text)["input_ids"]

print(tokens)
print(input_ids)
```

This code loads and uses the tokenizer. It does **not** load or execute the model itself, and it does not produce a sentiment prediction.

### Explicitly loading and running the model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("I love this course!", return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)
```

Here, `AutoModelForSequenceClassification.from_pretrained(...)` loads the trained neural network and its weights. Calling `model(**inputs)` runs the model and returns its outputs, including logits.

### Using a pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

result = classifier("I love this course!")
print(result)
```

A pipeline provides a convenient interface that coordinates tokenization, model execution, and task-specific output processing. Naming the model explicitly makes it clear which pretrained model is being used. If a task pipeline is created without a model name, it uses a task default associated with the installed library version; specifying a model is more explicit and reproducible.

**Remember:** `transformers` is the library; the tokenizer prepares the input; the model computes outputs; the pipeline coordinates these steps.

## 3. Tokenization and input IDs

A neural network does not directly consume ordinary text. The tokenizer maps text to a sequence of tokens, then maps each token to an integer ID from its vocabulary.

- **Token:** A unit of text. It can be a word, part of a word, punctuation, or a special token.
- **Token ID:** The integer associated with a token in a particular tokenizer's vocabulary.
- **Input IDs:** The sequence of token IDs supplied to the model.

Illustrative flow:

```text
"I love this course!"
        ↓
tokens
        ↓
token IDs
        ↓
input_ids
```

The exact token boundaries and IDs depend on the tokenizer. One written word does not always equal one token: a tokenizer may split a word into multiple subword pieces, and each piece receives its own ID.

Some tokenizers also add special tokens. In the DistilBERT sentiment tokenizer used in the example, `[CLS]` and `[SEP]` are special tokens. Example IDs such as `[101, 1045, 2293, 2023, 2607, 999, 102]` belong to that particular tokenizer; do not assume that another tokenizer uses the same IDs or special-token conventions.

Token IDs are vocabulary indices, not numerical measurements of semantic similarity. The model's embedding layer maps token IDs to learned vectors, which are then processed by the network.

## 4. Logits and softmax

### What are logits?

Logits are raw scores produced by a model. They are not automatically probabilities:

- They can be negative or positive.
- They do not have to lie between 0 and 1.
- They do not have to sum to 1.

For an illustrative binary sentiment classifier, suppose the logits are:

```text
NEGATIVE: -3.1
POSITIVE:  3.4
```

The higher positive logit indicates that the model scores the positive class more highly, but the raw number `3.4` does not mean a 340% probability.

### What does softmax do?

For a vector of logits, softmax converts each score into a normalized value between 0 and 1, with all output values summing to 1:

\[
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Here, \(z_i\) is the logit for class \(i\), and the denominator sums the exponentials of all class logits.

For logits `[-3.1, 3.4]`, softmax produces approximately:

```text
NEGATIVE: 0.0015  (0.15%)
POSITIVE: 0.9985  (99.85%)
```

The numbers are illustrative. They demonstrate that the higher logit gets the larger normalized probability.

In PyTorch, for classification logits, one can calculate probabilities like this:

```python
import torch

probabilities = torch.softmax(outputs.logits, dim=-1)
print(probabilities)
```

Softmax probabilities are not a guarantee that a prediction is correct, and the scores may not be perfectly calibrated. The appropriate post-processing depends on the model and task. In autoregressive text generation, logits typically score possible next tokens, and a decoding strategy is used to select a token.

## 5. Padding

Padding is an input-preparation technique. It is **not** a way to fill in or predict the next word.

When processing a batch, input sequences can have different lengths. For example:

```text
Sentence A: I love this course   (4 illustrative tokens)
Sentence B: I am happy           (3 illustrative tokens)
```

To create a rectangular batch, the shorter sequence can be padded:

```text
Sentence A: I   love  this  course
Sentence B: I   am    happy [PAD]
```

`[PAD]` is a placeholder token used to make the sequence lengths match. It does not mean the model should generate a missing word.

Padding is commonly useful when batching inputs. It may not be needed when processing one input at a time, and the actual number of tokens can differ because of special tokens and the tokenizer's segmentation.

Example:

```python
inputs = tokenizer(
    ["I love this course", "I am happy"],
    padding=True,
    return_tensors="pt"
)
```

- `padding=True` pads the sequences to a common length for this batch.
- `return_tensors="pt"` returns PyTorch tensors.
- The tokenizer typically handles the padding token and the corresponding mask.

## 6. Attention masks

An attention mask indicates which sequence positions are valid input and which are padding, using the common Hugging Face convention:

- `1` = a real token position.
- `0` = a padding position.

For the illustrative padded batch:

```text
Sentence A: I   love  this  course
Mask A:     1   1     1     1

Sentence B: I   am    happy [PAD]
Mask B:     1   1     1     0
```

The input IDs and attention mask have matching batch and sequence dimensions. The mask lets the model handle padding appropriately; padding positions are not simply deleted from the input tensor.

In the tokenizer example, inspect both outputs:

```python
inputs = tokenizer(
    ["I love this course", "Good"],
    padding=True,
    return_tensors="pt"
)

print(inputs["input_ids"])
print(inputs["attention_mask"])
```

The exact token IDs, padding ID, sequence lengths, and masks depend on the tokenizer and the text.

### Padding mask versus causal mask

These masks solve different problems:

- **Padding mask:** Identifies which positions are padding rather than real input tokens.
- **Causal mask:** Restricts which positions a token is allowed to attend to, preventing a token in an autoregressive language model from using future tokens when predicting the next token.

A causal mask is not the same as a padding mask. Depending on the architecture, a model may use both kinds of masking.

## 7. Putting the full classification flow together

```text
Raw text
   ↓
Tokenizer: text → tokens → input IDs
   ↓
Padding and attention mask, when needed
   ↓
Model: input IDs → learned embeddings → neural-network processing
   ↓
Logits
   ↓
Task-appropriate post-processing
   ↓
Prediction, such as a sentiment label and score
```

The steps are conceptually distinct:

- Tokenization and padding prepare the input **before** model execution.
- The model generates logits **during** model execution.
- Softmax is one possible post-processing step **after** logits are produced for a classification task.

A pipeline hides much of this sequence behind a single call, but the underlying stages still occur.

## 8. Common misconceptions to avoid

1. **“Transformers is the model.”**  
   No. Transformers is the library used to load and operate on models.

2. **“Loading a tokenizer runs the model.”**  
   No. Tokenizer-only code prepares text; a model must be loaded and called to perform neural-network inference.

3. **“A logit of 4 means a 400% probability.”**  
   No. Logits are raw scores. A suitable transformation, such as softmax for multiclass classification, can produce normalized probabilities.

4. **“Padding predicts the next word.”**  
   No. Padding aligns sequence lengths for processing. Next-token prediction is a different operation.

5. **“Attention masks and causal masks are the same.”**  
   No. Padding masks identify padding positions; causal masks restrict access to future positions.

6. **“The pipeline randomly changes models on each call.”**  
   No. A pipeline uses its configured model. When the model is not specified, a task default may be used; explicitly naming the model makes the choice clear.

7. **“A high probability guarantees the prediction is correct.”**  
   No. A model can be confident and still be wrong.

## 9. Quick self-check

Try answering these without looking back:

1. What is the difference between the Transformers library, a tokenizer, and a model?
2. What does `model(**inputs)` do that `tokenizer(text)` does not?
3. Why can one word produce multiple token IDs?
4. What are logits, and why is a logit not a probability?
5. What does softmax do to classification logits?
6. When and why might padding be applied?
7. What do `1` and `0` usually mean in a Hugging Face attention mask?
8. How does a causal mask differ from a padding mask?
9. What work does a sentiment-analysis pipeline coordinate?

## Week 2 takeaway

A pretrained classification model does not receive raw text directly. The tokenizer prepares numerical inputs, padding and masks support batch processing, the model computes logits, and task-appropriate post-processing turns those outputs into a prediction. Hugging Face Transformers provides the tools to load and use each component, while pipelines make the complete workflow easier to call.
