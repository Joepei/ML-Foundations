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

The middle tensor is the residual stream: a sequence of contextual vectors that gets refined by each transformer block.

## Overall Structure

A decoder-only transformer language model has four main stages:

1. Token embeddings turn token IDs into vectors.
2. Transformer blocks repeatedly mix information across the sequence and apply nonlinear transformations.
3. A final normalization stabilizes the output representation.
4. A linear output projection converts hidden states into vocabulary logits.

In code, the forward pass is compact:

```python
x = self.embedding(x)
for block in self.transformer_blocks:
    x = block(x, token_positions)
x = self.norm_pre_out(x)
x = self.output_embedding(x)
```

Each transformer block keeps the shape the same:

\[
(\text{batch}, \text{sequence}, d_\text{model})
\rightarrow
(\text{batch}, \text{sequence}, d_\text{model})
\]

The block changes the representation, not the tensor rank. Self-attention aggregates information across positions, and the feed-forward network applies a nonlinear transformation at each position.

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

## Pre-Norm Blocks

This implementation uses pre-norm transformer blocks:

```python
x = x + self.multi_head_attention(self.norm_1(x), token_positions=token_positions)
x = x + self.ff(self.norm_2(x))
```

The normalization happens before attention and before the feed-forward network. The residual path itself stays clean.

One intuition for pre-norm is gradient flow. If a block is written as:

\[
x_{l+1} = x_l + F(\operatorname{Norm}(x_l))
\]

then the derivative from \(x_{l+1}\) back to \(x_l\) always has an identity term from the residual connection. Even if the learned branch is poorly initialized or noisy, earlier layers can still receive signal through that direct path.

That is the meaning of the "residual stream": the model carries a direct representation forward while each block contributes an update.

## RMSNorm

RMSNorm stabilizes vector magnitude by dividing by the root mean square of activations, then restoring learned scale with a gain vector:

\[
\operatorname{RMSNorm}(x)
=
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}}
\odot g
\]

The intuition is that the division controls magnitude, while \(g\) gives the model flexibility to learn useful scaling.

The implementation casts to `float32` before computing the statistic:

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

## Feed-Forward Network

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

## Rotary Positional Embeddings

Self-attention has no built-in notion of token order. RoPE adds position information by rotating query and key vectors in paired dimensions.

The important implementation insight is that there is no need to construct a full rotation matrix \(R\). That matrix is mostly structure, and the operation is just pairwise rotation:

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

## Scaled Dot-Product Attention

Attention lets each token decide which earlier token representations to read from.

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

## Multi-Head Self-Attention

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

The head dimension is then treated like an extra batch dimension inside attention. After attention, the heads are concatenated again before the output projection:

```python
concat_attn = rearrange(attn, "batch h seq_len d -> batch seq_len (h d)")
return self.WO(concat_attn)
```

`rearrange` is a view-style reshape operation when possible. It does not change the math, and PyTorch autograd tracks gradients through it.

## Where RoPE Is Applied

RoPE is applied to queries and keys, not values:

```python
Q = self.rope(Q, token_positions)
K = self.rope(K, token_positions)
```

The reason is conceptual:

- \(Q \cdot K\) decides which tokens attend to which, so position should affect queries and keys.
- \(V\) stores the content to retrieve, so position should not rotate the content itself.

RoPE is applied after splitting into heads because each head has its own \(d_k\) dimension. The rotation should happen inside each head independently.

During training, token positions are usually:

```
[0, 1, ..., sequence_length - 1]
```

During inference with cached keys and values, token positions should reflect the actual position of the new token, not restart at zero.

## Full Transformer Block

The full block is:

```python
x = x + self.multi_head_attention(self.norm_1(x), token_positions=token_positions)
x = x + self.ff(self.norm_2(x))
```

The second residual connection must use the output of the first residual connection, not the original input. The feed-forward network transforms the representation after attention has already updated it.

For multiple blocks, use `nn.ModuleList` instead of a plain Python list:

```python
self.transformer_blocks = nn.ModuleList()
for _ in range(num_layers):
    self.transformer_blocks.append(Transformer_block(...))
```

Without `ModuleList`, PyTorch will not properly register the block parameters. That breaks things like `model.parameters()`, `model.train()`, `model.eval()`, and checkpointing.

## Implementation Checklist

When implementing a decoder-only transformer, I would check:

- Do token IDs become `(batch, sequence, d_model)` through embedding lookup?
- Are custom layers inheriting from `nn.Module` and calling `super().__init__()`?
- Are learnable tensors wrapped in `nn.Parameter`?
- Does RMSNorm keep dimensions for broadcasting and compute the statistic in `float32`?
- Does the pre-norm block preserve a clean residual path?
- Does the feed-forward network use the original `x` for both SwiGLU branches?
- Does RoPE avoid materializing the full rotation matrix?
- Is RoPE applied to `Q` and `K`, but not `V`?
- Are `WQ`, `WK`, and `WV` each one large projection before splitting heads?
- Is the causal mask lower triangular over `(query, key)` positions?
- Does the second residual connection use the attention-updated `x`?
- Are stacked blocks stored in `nn.ModuleList`?

The main intuition is that a transformer block keeps the tensor shape stable while repeatedly updating the representation. Attention moves information across positions; the feed-forward network transforms each position; residual connections preserve a path for information and gradients to flow.
