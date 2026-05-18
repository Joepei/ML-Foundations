# Tokenization and BPE

Tokenization is the preprocessing step that turns text into discrete units a model can embed. A language model does not operate directly on Python strings; it operates on integer token IDs that index learned embedding vectors.

## Text, Bytes, and Integers

A piece of text can be viewed at multiple levels:

- a string, such as `"hello"`
- bytes, such as `b"\x68\x65\x6c\x6c\x6f"`
- integer byte values, such as `(104, 101, 108, 108, 111)`

A byte has 8 bits, so it can represent 256 values. ASCII characters use one byte, while many non-ASCII characters use multiple bytes under UTF-8.

## Why Not Use Words Directly?

Word-level tokenization creates short sequences, but the vocabulary can become huge. It also handles typos, new words, and morphology poorly. Words like `run`, `running`, and `runner` receive independent entries even though they share structure.

Byte-level tokenization solves out-of-vocabulary problems because any text can be represented as bytes. The cost is longer sequences, which make training slower and increase memory use.

Subword tokenization is a compromise. It can represent arbitrary text while still learning frequent chunks that shorten sequences.

## Byte-Pair Encoding

Byte-Pair Encoding starts with individual bytes and repeatedly merges the most frequent adjacent pair into a new token. The result is a vocabulary that contains both small units and common larger units.

The algorithm can be summarized as:

1. Initialize the vocabulary with all byte values.
2. Split the corpus into pre-tokens.
3. Count adjacent token pairs within each pre-token.
4. Merge the most frequent pair.
5. Repeat until the desired vocabulary size is reached.

Pre-tokenization is important because it prevents merges from crossing boundaries that should remain separate, such as punctuation or whitespace boundaries.

## Implementation Detail

For large corpora, it is better to stream matches than to materialize every match at once. In Python, `re.finditer` returns an iterator and can be more memory efficient than `re.findall`.
