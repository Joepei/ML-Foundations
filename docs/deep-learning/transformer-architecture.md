# Transformer Architecture

A transformer language model maps token IDs to contextual representations, then projects those representations into logits over the vocabulary. During training, it learns to predict the next token at each position.

## Token Embeddings

The input begins as a batch of token IDs:

\[
(\text{batch}, \text{sequence})
\]

An embedding table maps each token ID to a vector:

\[
(\text{batch}, \text{sequence}, d_\text{model})
\]

In code, an embedding layer is essentially batched lookup:

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
```

## Pre-Norm Transformer Blocks

A pre-norm transformer block applies normalization before the attention and feed-forward sublayers. The residual path remains direct, which helps preserve gradient flow through deep stacks.

The key intuition is that the residual connection gives the gradient an identity path. Even if a learned transformation is initially poor, earlier layers can still receive useful signal.

## RMSNorm

RMSNorm stabilizes vector magnitude by dividing by the root mean square of activations, then restoring learned scale with a parameter vector:

\[
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}} \odot g
\]

An implementation detail: convert to `float32` before computing the normalization statistic for better numerical precision, then cast back to the input dtype.

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

## Attention

Self-attention lets each token representation aggregate information from other positions. In decoder-only language models, causal masking prevents a token from attending to future tokens.

Scaled dot-product attention computes compatibility between queries and keys, normalizes the scores, and uses them to average values:

\[
\operatorname{Attention}(Q, K, V) =
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Multi-head attention performs this operation in several learned subspaces, allowing the model to attend to different relationships in parallel.

## Full Language Model

A transformer language model stacks multiple transformer blocks, applies a final normalization, then projects to vocabulary logits. When storing blocks in PyTorch, use `nn.ModuleList` rather than a plain Python list so parameters are registered correctly.
