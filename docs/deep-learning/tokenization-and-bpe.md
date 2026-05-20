# Tokenization and Byte-Pair Encoding

Tokenization is the bridge between text and a language model. The model does not consume Python strings directly. It consumes integer token IDs, then uses those IDs to look up embedding vectors.

The basic chain is:

```
text -> token IDs -> embedding vectors
```

During training, each token gets its own learned embedding. A piece of text first becomes a list of \(n_\text{tokens}\) token IDs, then those IDs become a sequence of embedding vectors with shape roughly:

\[
n_\text{tokens} \times d_\text{embedding}
\]

The important design question is where to put the boundary between "raw text" and "model input."

Byte-level tokenization is the most universal option:

```
unhappiness -> (u, n, h, a, p, p, i, n, e, s, s)
            -> (117, 110, 104, 97, 112, 112, 105, 110, 101, 115, 115)
```

The benefit is that there are no out-of-vocabulary tokens because any text can be represented as bytes. The cost is sequence length: an emoji can take several bytes, and long token sequences slow down training, use more memory, and make long-range patterns harder to learn.

Word-level tokenization goes the other direction:

```
"happy unhappiness" -> ["happy", "unhappiness"]
```

This creates shorter sequences, but the vocabulary can become huge. It also handles typos, new words, and morphology poorly. Words like `run`, `running`, and `runner` get independent embeddings even though they clearly share structure.

Subword tokenization is the compromise:

```
unhappiness -> [un, happi, ness]
```

Its vocabulary can range from single bytes to full words. Random or unseen text can still be represented, while frequent pieces become compact tokens. Usually we preset a vocabulary size and let the training data decide which subwords are frequent enough to add.

## The Type Triangle

The part of byte-level BPE that felt easiest to get wrong was not the high-level algorithm. It was keeping the data types straight.

Three Python types show up constantly:

- `str`: decoded text, such as `"hello"`.
- `bytes`: raw UTF-8 bytes, such as `b"\x68\x65\x6c\x6c\x6f"`.
- `int`: a byte value or token ID, such as `104`.

Some useful conversions:

```python
s.encode("utf-8")       # str -> bytes
b.decode("utf-8")       # bytes -> str
tuple(b"hi")            # bytes -> tuple[int], gives (104, 105)
bytes([104])            # int -> one-byte bytes object, gives b"h"
```

The key implementation detail is that pre-token sequences should be stored as tuples of integer token IDs, not as `bytes` objects. A single byte can only represent values from 0 to 255. After BPE starts adding merged tokens, token IDs become 256, 257, and so on, so `bytes([256])` fails. A tuple of integers has no such limit.

```python
tuple(b"hi")      # (104, 105), useful as the initial pre-token form
bytes([256])      # ValueError
```

The vocabulary and merges have separate roles:

```python
vocab:  dict[int, bytes]           # token ID -> byte string
merges: list[tuple[bytes, bytes]]  # ordered merge pairs
```

That means special tokens and merged tokens should also be stored as bytes:

Wrong: `self.vocab[token_id] = "<|endoftext|>"` stores text. Right: `self.vocab[token_id] = "<|endoftext|>".encode("utf-8")` stores the bytes the tokenizer actually operates on.

And merges should store the byte values associated with token IDs, not the integer IDs themselves:

Wrong: `self.merges.append(max_key)` stores token IDs. Right: `self.merges.append((self.vocab[max_key[0]], self.vocab[max_key[1]]))` stores the byte strings being merged.

## Training Byte-Level BPE

Byte-Pair Encoding starts with the 256 possible byte values, then repeatedly merges the most frequent adjacent pair into a new token. The goal is compression: frequent adjacent pieces become single tokens, which shortens the tokenized sequence.

At a high level:

1. Initialize the vocabulary with byte tokens `0` through `255`.
2. Add special tokens, such as `<|endoftext|>`, as indivisible vocabulary entries.
3. Split the corpus into pre-tokens.
4. Count adjacent token pairs inside those pre-tokens.
5. Pick the most frequent pair.
6. Add a new token whose bytes are the concatenation of that pair.
7. Replace occurrences of that pair with the new token ID.
8. Repeat until the target vocabulary size is reached.

The new token must be represented by the new token ID, not by adding the two old IDs numerically:

Wrong: `new_key.append(k[i] + k[i + 1])` turns two IDs into an unrelated integer, such as `104 + 101 = 205`. Right: `new_key.append(new_token_id)` appends the ID assigned to the merged token.

The vocabulary entry for that new token is the concatenation of the underlying bytes:

```python
self.vocab[new_id] = self.vocab[max_key[0]] + self.vocab[max_key[1]]
```

If two pairs have the same frequency, this implementation prefers the lexicographically greater pair of byte strings:

```python
max(self.pairs, key=lambda x: (self.pairs[x], self.vocab[x[0]], self.vocab[x[1]]))
```

### Pre-Tokenization

Pre-tokenization decides which spans BPE is allowed to merge inside. If BPE directly merged frequent consecutive bytes across the raw corpus, it could be both expensive and too sensitive to punctuation. For example, `dog.` and `dog!` might become separate tokens even though the useful shared piece is mostly `dog`.

The pattern used here groups contractions, optional-leading-space words and numbers, punctuation-like runs, and whitespace:

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

The important consequence is that BPE learns merges within these pre-token spans instead of freely merging across every boundary in the corpus.

For large files, `re.finditer` is a better fit than `re.findall` because it streams matches one at a time instead of materializing every match into a list:

```python
for m in re.finditer(PAT, part):
    pre_tokens[tuple(m.group().encode("utf-8"))] += 1
```

### Special Tokens

Special tokens should be added to the vocabulary, but they should not be broken apart and fed through ordinary BPE pre-tokenization. For a token like `<|endoftext|>`, the tokenizer should emit one special-token ID, not several IDs for `<`, `|`, `end`, and so on.

The text is split on special tokens before applying the pre-tokenization regex:

```python
pattern = "|".join(re.escape(t) for t in special_tokens)
parts = re.split(pattern, text)
```

When encoding later, it is useful to keep the delimiters:

```python
parts = re.split(f"({pattern})", text)
```

The capturing group keeps the special tokens in the result so the encoder can emit their ID directly. If special tokens can overlap, sort them longest-first before building the regex; Python regex alternation checks alternatives from left to right, so a shorter token can otherwise consume the beginning of a longer one.

### File Chunks and Line Endings

For pre-tokenizing larger files, the implementation reads binary chunks and aligns chunk boundaries on `<|endoftext|>`. Binary mode is useful because `seek()` works with exact byte offsets:

```python
with open(input_path, "rb") as f:
    f.seek(start)
    text = f.read(end - start).decode("utf-8")
```

One subtle bug is that binary mode preserves platform-specific line endings. On Windows, a file may contain `\r\n`, while a reference snapshot or text-mode read may normalize this to `\n`. That difference changes the pre-tokens and therefore the learned merges.

The fix is to normalize after decoding:

```python
text = (
    f.read(end - start)
    .decode("utf-8")
    .replace("\r\n", "\n")
    .replace("\r", "\n")
)
```

The order matters: replace `\r\n` before replacing standalone `\r`, otherwise one Windows newline can accidentally become two newlines.

### Updating Pair Counts

The naive implementation would recompute all adjacent pair counts after every merge. That is simple, but expensive. This implementation keeps two related structures:

```python
self.pairs      # pair -> count
self.pair_index # pair -> set of pre-token sequences containing that pair
```

When a pair is merged, only pre-tokens containing that pair can change. The `pair_index` lets the algorithm update affected pre-tokens and their neighboring pair counts instead of starting from the full corpus every time.

There is a Python detail hiding here too: do not mutate a dictionary while iterating over it. If the shape of a dictionary needs to change during an update, iterate over a copy or build a replacement structure. Otherwise Python raises `RuntimeError: dictionary changed size during iteration`.

### Multiprocessing as an Engineering Detail

Pre-tokenization can be parallelized because chunks can be counted independently before their dictionaries are merged. On Windows, multiprocessing has a few sharp edges: worker functions need to live at module top level, workers do not share `self.pre_tokens`, and scripts that start multiprocessing should guard entry code with `if __name__ == "__main__":`.

This is an implementation detail, not the essence of BPE. The core algorithm is still sequential at the merge step because each merge changes the state used to choose the next merge. That is also why BPE training is not naturally GPU-friendly: it relies on ordered updates and dynamic hash maps more than dense tensor operations.

## Encoding

Training produces a vocabulary and an ordered list of merges. Encoding applies that learned merge order to new text.

The encoding flow is:

1. Split on special tokens, keeping them as their own parts.
2. For normal text parts, run the same pre-tokenization pattern.
3. Convert each byte into its initial token ID.
4. Repeatedly apply the highest-priority learned merge that appears in the pre-token.
5. Flatten the result into `list[int]`.

Merge priority matters. The encoder should not simply merge left-to-right. It should find the adjacent pair with the lowest rank in the learned merge list, apply that merge, and repeat until no learned pair remains.

One subtle point is that initial bytes should be converted through the vocabulary, not assumed to equal their ASCII values:

The fragile version is `tuple(m.group().encode("utf-8"))`, which gives raw byte values. The safer version maps each byte through the vocabulary:

```python
tuple(self.reverse_vocab[bytes([b])] for b in m.group().encode("utf-8"))
```

In this tokenizer, the initial vocabulary maps `i` to `bytes([i])`, so the byte value and token ID happen to match for the first 256 entries. But that is an implementation choice, not a universal truth. An externally trained vocabulary could assign different IDs.

## Decoding

Decoding goes the other direction:

```
token IDs -> bytes -> UTF-8 text
```

The tokenizer looks up each ID in `vocab`, concatenates the bytes, then decodes the result:

```python
return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")
```

Using `errors="replace"` makes decoding more robust if the byte sequence is invalid UTF-8. Instead of crashing, Python inserts the replacement character.

## Saving the Tokenizer

The trained tokenizer needs both `vocab` and `merges`.

The vocabulary is useful to inspect, so JSON is convenient, but JSON cannot store bytes directly and requires dictionary keys to be strings. A lossless workaround is to decode bytes with `latin-1` on save and encode with `latin-1` on load:

```python
json.dump({str(k): v.decode("latin-1") for k, v in vocab.items()}, f)
vocab = {int(k): v.encode("latin-1") for k, v in json.load(f).items()}
```

`latin-1` maps byte values 0 through 255 one-to-one into Unicode code points, so it preserves arbitrary bytes. UTF-8 is not a safe substitute here because not every byte sequence is valid UTF-8.

The merges are naturally ordered Python tuples of bytes, so pickle is a simple fit:

```python
pickle.dump(merges, f)
merges = pickle.load(f)
```

In practice, `vocab` as JSON and `merges` as pickle is a useful split: one is easy to inspect while debugging, and the other preserves the exact Python structure.

## Training Notes

Tokenizer training belongs with the tokenizer, not the transformer training loop. In the TinyStories experiments, the tokenizer vocabulary size was part of the setup: train a tokenizer, encode the dataset, then train the model on token IDs.

One useful experiment is comparing compression across tokenizers and datasets. For example:

```
bytes per token = total UTF-8 bytes / total tokens
```

This makes the tokenizer quality visible in a practical way. A tokenizer trained on one distribution may compress that distribution better than another one. Better compression usually means shorter token sequences, which affects training cost and context usage downstream.

## Implementation Checklist

When implementing a byte-level BPE tokenizer, I would check:

- Are special tokens stored as bytes and kept out of ordinary BPE merges?
- Are pre-token sequences represented as tuples of token IDs rather than bytes?
- Are merged vocab entries built by concatenating bytes?
- Are merge records stored as `tuple[bytes, bytes]`, not token ID pairs?
- Are Windows line endings normalized if binary chunk reads are used?
- Does pre-tokenization stream matches with `finditer` for large corpora?
- Does encoding apply learned merges by priority rather than left-to-right?
- Are initial bytes mapped through `reverse_vocab` instead of assuming byte value equals token ID?
- Are vocab JSON conversions using `latin-1` rather than UTF-8?
- Is the tokenizer evaluated with both correctness checks and compression behavior?

The main intuition is simple but important: tokenization is not just a preprocessing detail. It decides the discrete units the model gets to see. Subword tokenization works because it keeps the representation open-ended like bytes, while letting frequent patterns become compact learned units.
