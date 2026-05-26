# Transformer Architecture

A transformer language model takes a batch of token IDs and predicts the next token at every position.

During training:

```
input:  batched sequence of integer token IDs
output: logits over the vocabulary at each position
loss:   cross entropy against the actual next token
```

During inference, the model generates one token, appends it to the input sequence, and repeats.

The high-level shape flow is:

\[
(\text{batch}, \text{sequence})
\rightarrow
(\text{batch}, \text{sequence}, d_\text{model})
\rightarrow
(\text{batch}, \text{sequence}, \text{vocab})
\]


## Overall Architecture

A decoder-only transformer language model has five main stages:

1. Token embeddings turn token IDs into vectors.
2. A stack of transformer blocks runs `num_layers` times, repeatedly mixing information across the sequence and applying nonlinear transformations.
3. A final pre-output normalization stabilizes the last residual-stream representation.
4. A linear output projection converts hidden states into vocabulary logits.
5. A final softmax converts logits into probabilities when the model needs a next-token distribution.

The full model flow is:

```
token IDs
(batch, sequence)
   |
   v
token embeddings
(batch, sequence, d_model)
   |
   v
+--------------------------------------------------------------+
| transformer block, repeated num_layers times                 |
|                                                              |
| x -> RMSNorm -> multi-head causal attention -> add to x       |
| x -> RMSNorm -> SwiGLU feed-forward network -> add to x       |
+--------------------------------------------------------------+
   |
   v
final pre-output RMSNorm
(batch, sequence, d_model)
   |
   v
final Linear layer
(batch, sequence, vocab_size logits)
   |
   v
softmax over vocab dimension
(batch, sequence, vocab_size probabilities)
```

In code, the forward pass is compact:

```python
x = self.embedding(x)
for block in self.transformer_blocks:
    x = block(x, token_positions)
x = self.norm_pre_out(x)
x = self.output_embedding(x)
```

The model usually returns logits from the forward pass. Softmax is applied when probabilities are needed, for example during sampling:

```python
probs = torch.softmax(logits, dim=-1)
```

## PyTorch Modules

The implementation uses custom modules for pieces like `Linear`, `Embedding`, `RMSNorm`, attention, and the transformer block. Inheriting from `nn.Module` matters because PyTorch then knows how to manage the model.

The practical benefits:

- `nn.Parameter` values are registered as learnable parameters.
- Nested submodules are tracked automatically.
- `model.to("cuda")` moves registered parameters and buffers to the GPU.
- `state_dict()` and `load_state_dict()` work cleanly.
- Calling `model(x)` uses the module's `forward()` method.

The small but important ritual is calling `super().__init__()` before assigning parameters or submodules.

## Token Embeddings

The input begins as integer token IDs:

\[
(\text{batch}, \text{sequence})
\]

An embedding table maps each token ID to a learned vector:

\[
(\text{batch}, \text{sequence}, d_\text{model})
\]

In code, an embedding layer is batched lookup:

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
```

This is different from a linear layer. A linear layer mixes input dimensions with a matrix multiply. An embedding layer indexes rows from a learned table.

## Components Inside the Block

The internal flow of a block is:

1. Normalize the residual stream.
2. Run causal multi-head self-attention.
3. Add the attention output back to the residual stream.
4. Normalize the updated residual stream.
5. Run the feed-forward network.
6. Add the feed-forward output back to the residual stream.

### Pre-Norm and RMSNorm

This implementation uses pre-norm transformer blocks:

```python
x_norm = self.norm_1(x)
x = x + self.multi_head_attention(x_norm, token_positions=token_positions)

x_norm = self.norm_2(x)
x = x + self.ff(x_norm)
```

The normalization happens before attention and before the feed-forward network. The residual path itself stays clean.

One intuition for pre-norm is gradient flow. If a block is written as:

\[
x_{l+1} = x_l + F(\operatorname{Norm}(x_l))
\]

then the derivative from \(x_{l+1}\) back to \(x_l\) always has an identity term from the residual connection. Even if the learned branch is poorly initialized or noisy, earlier layers can still receive signal through that direct path.

That is the meaning of the "residual stream": the model carries a direct representation forward while each block contributes an update.

RMSNorm is applied to one token-position activation at a time. For a single activation vector:

\[
a = [a_1, a_2, \ldots, a_{d_\text{model}}]
\]

the root mean square is:

\[
\operatorname{RMS}(a)
=
\sqrt{
\frac{1}{d_\text{model}}
\sum_{i=1}^{d_\text{model}} a_i^2
+ \epsilon
}
\]

Each coordinate is then normalized and rescaled:

\[
\operatorname{RMSNorm}(a_i)
=
\frac{a_i}{\operatorname{RMS}(a)}
\cdot g_i
\qquad
i = 1, \ldots, d_\text{model}
\]

The intuition is that the division controls the magnitude of the activation vector at that one token position, while the learned gain vector:

\[
g = [g_1, g_2, \ldots, g_{d_\text{model}}]
\]

gives the model \(d_\text{model}\) learned scale values in total. There is one \(g_i\) for each hidden coordinate, and the same gain vector is reused at every batch and sequence position.

Raw implementation:

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.W).to(in_dtype)
```

Two implementation details matter:

- `keepdim=True` keeps the last dimension as size 1, so division broadcasts back over \(d_\text{model}\).
- Computing the RMS statistic in `float32` is more stable for lower precision training.

### Attention Sublayer

The first sublayer is attention:

```python
x_norm = self.norm_1(x)
x = x + self.multi_head_attention(x_norm, token_positions=token_positions)
```
Attention, in its simplest form, is a weighted average of value vectors. The weights are determined by how **compatible (similar)** the current token’s query is to each other token’s key. The values contain the information that gets mixed into the current token’s representation.

In a decoder-only transformer, this attention is causal, so position \(i\) can use tokens up to \(i\), but not tokens after \(i\).

The attention sublayer internally does this:

1. Project the normalized input into queries, keys, and values.
2. Split those projections into multiple heads.
3. Apply RoPE to queries and keys.
4. Compute masked scaled dot-product attention.
5. Concatenate heads and project back to \(d_\text{model}\).

#### Multi-Head Self-Attention

The intuition behind multiple heads is that the model may need to attend to the same context for different reasons. One head might focus on the previous noun that a pronoun refers to, another might track local syntax, and another might look for longer-range topic or formatting cues. The heads are not guaranteed to become clean human-labeled linguistic modules, but splitting attention into heads gives the model separate subspaces where different relationships can be represented at the same time.

Multi-head attention uses four linear layers:

- `WQ`: query projection
- `WK`: key projection
- `WV`: value projection
- `WO`: output projection

The implementation does not create one projection layer per head. Instead, `WQ`, `WK`, and `WV` each project to `num_heads * d_k` in one matrix multiply, then the head dimension is split out:

```python
Q = rearrange(Q, "batch seq_len (h d_k) -> batch h seq_len d_k", h=self.num_heads)
K = rearrange(K, "batch seq_len (h d_k) -> batch h seq_len d_k", h=self.num_heads)
V = rearrange(V, "batch seq_len (h d_v) -> batch h seq_len d_v", h=self.num_heads)
```

This is primarily for computation efficiency. GPUs prefer fewer large matrix operations over many small ones.

#### Rotary Positional Embeddings

Self-attention has no built-in notion of token order. RoPE adds position information by rotating query and key vectors in paired dimensions.

For each position \(m\), RoPE applies a rotation \(R_m\) to the query and key vectors at that position:

\[
q_m^\prime = R_m q_m,
\qquad
k_n^\prime = R_n k_n
\]

Each two-dimensional coordinate pair gets rotated by a position-dependent angle. For one pair \((x_{2i}, x_{2i+1})\), the operation is:

\[
\begin{bmatrix}
x_{2i}^\prime \\
x_{2i+1}^\prime
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
\]

After RoPE, the dot product between a query at position \(m\) and a key at position \(n\) includes information about their distance \(m - n\). This lets attention distinguish “the previous token,” “a token 5 steps back,” or “a token much earlier,” instead of only comparing token content.

An important implementation insight is that there is no need to construct a full rotation matrix \(R\). That matrix is mostly structure, and the operation is just pairwise rotation:

```python
x_even = x[..., ::2].clone()
x_odd = x[..., 1::2].clone()
x[..., ::2] = x_even * cos - x_odd * sin
x[..., 1::2] = x_even * sin + x_odd * cos
```

This is a common deep learning pattern: understand what the matrix does, then compute the effect directly if materializing the matrix would waste memory and time.

The sine and cosine buffers come from broadcasting inverse frequencies against sequence positions. The target shape is:

\[
(\text{max sequence length}, d_k / 2)
\]

so positions are shaped like a column and inverse frequencies like a row:

```python
inv_freq = inv_freq.unsqueeze(0)
positions = torch.arange(max_seq_len).unsqueeze(1)
sin = torch.sin(positions * inv_freq)
cos = torch.cos(positions * inv_freq)
```

The buffers can be registered with `persistent=False` so they move with the module device but do not need to be saved in the checkpoint.


RoPE is applied to queries and keys, not values:

```python
Q = self.rope(Q, token_positions)
K = self.rope(K, token_positions)
```

The reason is conceptual:

- \(Q \cdot K\) decides which tokens attend to which, so position should affect queries and keys.
- \(V\) stores the content to retrieve, so position should not rotate the content itself.

RoPE is applied after splitting into heads because each head has its own \(d_k\) dimension. The rotation should happen inside each head independently.


#### Scaled Dot-Product Attention

The core formula is:

\[
\operatorname{Attention}(Q, K, V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

The query-key product determines which positions attend to which. The values contain the content that gets aggregated.

In a decoder-only language model, causal masking prevents future-token leakage. At position \(i\), the model can see tokens up to \(i\), but not tokens after \(i\). In code, this is a lower-triangular mask:

```python
mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
```

It is lower triangular because the attention score matrix is indexed as:

```
(query position, key position)
```

The first query row can only keep the first key, the second row can keep the first two keys, and so on.

The head dimension is then treated like an extra batch dimension inside attention. After attention, the heads are concatenated again before the output projection:

```python
concat_attn = rearrange(attn, "batch h seq_len d -> batch seq_len (h d)")
return self.WO(concat_attn)
```

`rearrange` is a view-style reshape operation when possible. It does not change the math, and PyTorch autograd tracks gradients through it.


### Feed-Forward Sublayer

After attention updates the residual stream, the block runs a second pre-norm sublayer and enters the feed-forward sublayer:

```python
x_norm = self.norm_2(x)
x = x + self.ff(x_norm)
```

The feed-forward network transforms each position independently. Attention mixes information across sequence positions; the feed-forward network then applies a nonlinear transformation to the representation at each position.

The original transformer used two linear transformations with a ReLU activation between them. Modern language models often use a gated feed-forward network instead.

This implementation uses SwiGLU:

\[
\operatorname{SwiGLU}(x)
=
W_2\left(\operatorname{SiLU}(W_1 x) \odot W_3 x\right)
\]

where:

\[
\operatorname{SiLU}(z) = z \cdot \sigma(z)
\]

SiLU is similar in spirit to ReLU because it keeps positive values and suppresses negative values, but it changes smoothly around zero instead of having a hard corner. That smoothness gives the feed-forward network a gentler nonlinear transition.

The gated part is also useful for gradient flow. A gated linear unit keeps a content branch, `W3(x)`, and modulates it with a learned gate, `SiLU(W1(x))`. Intuitively, the model gets a mostly linear path for information and gradients through the content branch, while the gate still gives it nonlinear control over which features should be amplified or suppressed.

In code:

```python
gate = self.W1(x)
return self.W2(gate * torch.sigmoid(gate) * self.W3(x))
```

The two branches have different roles. `W1` produces the gate, `W3` produces the content branch, and `W2` projects back to \(d_\text{model}\). A useful intuition is that the gate controls which features pass through while still preserving a nonlinear transformation.

The hidden dimension is often chosen around:

\[
d_\text{ff} \approx \frac{8}{3} d_\text{model}
\]

and rounded to a hardware-friendly multiple such as 64.

## Full Transformer Block

The full block is:

```python
x_norm = self.norm_1(x)
x = x + self.multi_head_attention(x_norm, token_positions=token_positions)

x_norm = self.norm_2(x)
x = x + self.ff(x_norm)
```

The second residual connection must use the output of the first residual connection, not the original input. The feed-forward network transforms the representation after attention has already updated it.

For multiple blocks, use `nn.ModuleList` instead of a plain Python list:

```python
self.transformer_blocks = nn.ModuleList()
for _ in range(num_layers):
    self.transformer_blocks.append(Transformer_block(...))
```

Without `ModuleList`, PyTorch will not properly register the block parameters. That breaks things like `model.parameters()`, `model.train()`, `model.eval()`, and checkpointing.

## Final Pre-Output Norm and Projection

After the last transformer block, the model still has internal activations shaped like:

\[
(\text{batch size}, \text{sequence length}, d_\text{model})
\]

The final pre-output normalization cleans up that residual stream before the vocabulary projection:

```python
x = self.norm_pre_out(x)
x = self.output_embedding(x)
```

That last linear layer maps each internal activation vector from \(d_\text{model}\) back to vocabulary logits:

\[
(\text{batch size}, \text{sequence length}, d_\text{model})
\rightarrow
(\text{batch size}, \text{sequence length}, \text{vocab size})
\]

So the output contains `batch_size * sequence_length * vocab_size` logits: one vocabulary-sized prediction for every token position in every sequence in the batch.

## Implementation Checklist

When implementing a decoder-only transformer, I would check:

- Do token IDs become `(batch, sequence, d_model)` through embedding lookup?
- Are custom layers inheriting from `nn.Module` and calling `super().__init__()`?
- Are learnable tensors wrapped in `nn.Parameter`?
- Does RMSNorm keep dimensions for broadcasting and compute the statistic in `float32`?
- Does the feed-forward network use the original `x` for both SwiGLU branches?
- Does RoPE avoid materializing the full rotation matrix?
- Is RoPE applied to `Q` and `K`, but not `V`?
- Are `WQ`, `WK`, and `WV` each one large projection before splitting heads?
- Is the causal mask lower triangular over `(query, key)` positions?
- Does the second residual connection use the attention-updated `x`?
- Are stacked blocks stored in `nn.ModuleList`?
- Does the final linear layer produce `(batch, sequence, vocab)` logits?

The main intuition is that a transformer block keeps the tensor shape stable while repeatedly updating the representation. Attention moves information across positions; the feed-forward network transforms each position; residual connections preserve a path for information and gradients to flow.
