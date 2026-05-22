# Resource Accounting for Transformer Training

Resource accounting is the habit of turning a model shape into concrete memory and compute numbers before a training run. It is easy to say "GPT-2 XL sized model" and miss the fact that the batch size, context length, vocabulary projection, optimizer state, and attention matrix all show up in different ways.

The useful pattern is to separate fixed costs from batch-dependent costs:

\[
M(B) = aB + b
\]

where \(B\) is batch size, \(aB\) is mostly activation memory, and \(b\) is mostly parameters plus training state. For FLOPs, the same kind of bookkeeping keeps the big matrix multiplications from getting lost inside the transformer block.

## GPT-2 XL Shape

I use a GPT-2 XL-shaped model throughout this page:

| Symbol | Meaning | Value |
|---|---:|---:|
| \(V\) | vocabulary size | 50,257 |
| \(S\) | context length | 1,024 |
| \(L\) | number of transformer layers | 48 |
| \(D\) | model dimension | 1,600 |
| \(H\) | attention heads | 25 |
| \(d_k\) | head dimension | 64 |
| \(d_{\text{ff}}\) | feed-forward hidden dimension | 6,400 |

The simplifying assumptions are:

- tensors are stored in `float32`, so each scalar uses 4 bytes
- token embeddings and output embeddings are tied for parameter counting
- biases, softmax cost, activation-function cost, residual additions, and normalization FLOPs are ignored
- dense causal attention computes the full \(S \times S\) attention pattern, then masks invalid positions
- one multiply-add counts as 2 FLOPs

These assumptions are intentionally clean. Real training runs can change the numbers with mixed precision, fused kernels, activation checkpointing, FlashAttention, optimizer sharding, tensor parallelism, memory fragmentation, and framework-specific saved tensors.

## AdamW Memory Accounting

For a GPT-style model with tied embeddings, the embedding parameters are:

\[
N_{\text{embed}} = VD + SD
\]

Each transformer block has:

- attention projections: \(4D^2\) for \(W_Q, W_K, W_V, W_O\)
- feed-forward weights: \(D(4D) + (4D)D = 8D^2\)
- two RMSNorm scale vectors: \(2D\)

So one layer contributes:

\[
N_{\text{layer}} = 12D^2 + 2D
\]

With one final RMSNorm, the total parameter count is:

\[
N_{\text{param}} = VD + SD + L(12D^2 + 2D) + D
\]

For the GPT-2 XL-shaped model:

\[
N_{\text{param}} = 1{,}556{,}764{,}800
\]

AdamW stores four parameter-sized tensors during training:

- parameters
- gradients
- first moment estimate \(m\)
- second moment estimate \(v\)

That gives the fixed memory term:

\[
b = 16N_{\text{param}}
\]

\[
b = 24{,}908{,}236{,}800 \text{ bytes} \approx 24.91 \text{ GB}
\]

### Activation Memory

Activation memory is batch-dependent because each example in the batch creates its own saved forward values.

For one transformer block, using a simplified saved-activation list:

| Component | Activation elements |
|---|---:|
| first RMSNorm | \(BSD\) |
| QKV projections | \(3BSD\) |
| attention scores | \(BHS^2\) |
| softmax attention weights | \(BHS^2\) |
| weighted sum of values | \(BSD\) |
| output projection | \(BSD\) |
| second RMSNorm | \(BSD\) |
| first feed-forward matrix | \(4BSD\) |
| SiLU | \(4BSD\) |
| second feed-forward matrix | \(BSD\) |

The \(BSD\) terms add to \(16BSD\), so one block stores:

\[
N_{\text{act block}} = 16BSD + 2BHS^2
\]

Across all layers, plus final RMSNorm, logits, and cross entropy:

\[
N_{\text{act}} =
L(16BSD + 2BHS^2) + BSD + 2BSV
\]

Factoring out batch size:

\[
N_{\text{act}} =
B\left[L(16SD + 2HS^2) + SD + 2SV\right]
\]

For this model, activation elements per batch element are:

\[
3{,}879{,}438{,}336
\]

So the batch-dependent memory term is:

\[
a = 4 \times 3{,}879{,}438{,}336
= 15{,}517{,}753{,}344 \text{ bytes}
\approx 15.52 \text{ GB}
\]

The final simplified memory expression is:

\[
M(B) = 15{,}517{,}753{,}344B + 24{,}908{,}236{,}800
\]

or:

\[
M(B) \approx 15.52B + 24.91 \text{ GB}
\]

On an 80 GB GPU:

\[
15.52B + 24.91 \leq 80
\]

\[
B \leq 3.55
\]

So the largest integer batch size under this simplified accounting is:

\[
\boxed{B_{\max} = 3}
\]

The main lesson is that AdamW state is a large fixed cost, but the activations dominate quickly as batch size increases.

## Forward-Pass FLOP Accounting

Now estimate one forward pass for one 1,024-token sequence. The tensor entering a transformer block has shape:

\[
S \times D
\]

For GPT-2 XL:

\[
1024 \times 1600
\]

### Attention

The Q, K, and V projections each map \(S \times D \rightarrow S \times D\). One projection costs \(2SD^2\), so all three cost:

\[
3 \cdot 2SD^2 = 15.73\text{B FLOPs}
\]

Attention scores are computed as \(QK^\top\). For one head this costs \(2S^2d_k\), and across \(H\) heads:

\[
H \cdot 2S^2d_k = 2S^2D = 3.36\text{B FLOPs}
\]

The weighted sum \(\text{Attention} \cdot V\) has the same cost:

\[
2S^2D = 3.36\text{B FLOPs}
\]

Then the concatenated attention output is projected through \(W_O\):

\[
2SD^2 = 5.24\text{B FLOPs}
\]

The easy mistake is to use \(d_k = 64\) and forget to multiply by 25 heads. Equivalently, use the full model dimension \(D = Hd_k = 1600\) for the all-head attention cost.

### Feed-Forward Network

The feed-forward hidden dimension is \(d_{\text{ff}} = 4D = 6400\). The two matrix multiplies are:

\[
S \times D \rightarrow S \times d_{\text{ff}}
\]

and:

\[
S \times d_{\text{ff}} \rightarrow S \times D
\]

Each costs:

\[
2SDd_{\text{ff}} = 20.97\text{B FLOPs}
\]

### Per-Layer Total

| Component | FLOPs |
|---|---:|
| Q, K, V projections | 15.73B |
| attention scores \(QK^\top\) | 3.36B |
| attention weighted sum | 3.36B |
| attention output projection | 5.24B |
| feed-forward first matrix | 20.97B |
| feed-forward second matrix | 20.97B |
| total per layer | 69.63B |

Across 48 layers:

\[
48 \times 69.63\text{B} \approx 3.34\text{T FLOPs}
\]

The final vocabulary projection maps:

\[
S \times D \rightarrow S \times V
\]

with cost:

\[
2SDV = 164.68\text{B FLOPs}
\]

That projection is done once at the end, not once per transformer layer.

So the full forward pass is:

\[
\text{transformer layers} + \text{final vocabulary projection}
\]

\[
\approx 3.34\text{T} + 0.165\text{T}
\]

\[
\boxed{3.51\text{T FLOPs}}
\]

## Accounting Checklist

When doing this kind of estimate, I try to check the same failure points every time:

- Are embeddings tied or untied?
- Are optimizer states included, especially AdamW's \(m\) and \(v\)?
- Are gradients counted separately from parameters?
- Is activation memory separated from fixed parameter memory?
- Is the attention term using all heads, not just one \(d_k\)-sized head?
- Is the attention output projection \(W_O\) included?
- Is \(d_{\text{ff}} = 4D\) or some architecture-specific value?
- Is the final vocabulary projection counted once, outside the layer loop?
- Is causal masking treated as dense attention or as an optimized triangular computation?
- Are the precision assumptions explicit?
