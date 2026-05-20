# GPU Systems and Utilization

GPU performance is not just about putting a model on CUDA. A training run can have all the model weights on the GPU and still be slow if the surrounding computation does not keep the hardware busy.

The useful mental model is:

```
Python training loop -> tensor operations -> CUDA kernels -> memory movement + arithmetic
```

GPUs are throughput machines. They are built to run many operations in parallel, especially dense tensor operations such as matrix multiplication. The implementation goal is to give the GPU large, regular chunks of work and avoid patterns that leave most of the hardware waiting.

## Memory Hierarchy

A GPU has a hierarchy of memory. The exact names and sizes depend on the hardware, but the broad pattern is:

```
global / high-bandwidth memory -> L2 / L1 caches -> shared memory / registers -> arithmetic units
```

Global memory is large but relatively slow. Shared memory and registers are much faster, but much smaller. A lot of GPU performance work is about reducing expensive global memory traffic and reusing values once they have been moved closer to the compute units.

That is why "same math" does not always mean "same speed." Two implementations can compute the same output while moving very different amounts of data.

## SIMT and Branch Divergence

GPUs operate in a SIMT model: single instruction, multiple threads. On NVIDIA GPUs, a warp is a group of threads that execute together, commonly 32 threads. Threads in the same warp execute the same instruction at the same time.

This makes branching expensive when neighboring threads take different paths. If some threads in a warp enter the `if` branch and others enter the `else` branch, the GPU cannot truly run both paths independently at the same time. It runs one path with some threads masked off, then runs the other path with the opposite threads masked off.

The idle masked threads are wasted work. This is branch divergence.

In practice, GPU code often prefers arithmetic masking over explicit branching. Instead of making threads choose separate control-flow paths, it can multiply by masks and keep the execution pattern more uniform.

## Low Precision Computation

Low precision helps in two ways.

First, fewer bits means fewer bits to move:

```
global memory -> caches / registers -> arithmetic units
```

If activations or weights use fewer bytes, the system can move more values through the memory hierarchy for the same bandwidth.

Second, lower precision can be faster to compute. Tensor cores are optimized for low-precision matrix multiplication, so operations in formats such as `float16` or `bfloat16` can be much faster than full `float32` matrix multiplication.

The caveat is that not every operation should be blindly downcast. Some reductions and normalization statistics are more sensitive. For example, RMSNorm is safer when the statistic is computed in `float32` and then cast back:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
return (x / rms * self.gain).to(in_dtype)
```

Quantization has a similar tradeoff. It can reduce memory and speed up some operations, but the benefit can be diluted if the implementation has to repeatedly dequantize or move through extra conversion steps.

## Operator Fusion

Every separate GPU operation has overhead. If a sequence of small operations each launches its own CUDA kernel, the GPU may spend too much time on launch overhead and memory traffic.

Operator fusion combines multiple operations into one kernel. Instead of reading from memory, computing one small step, writing an intermediate result, and then reading it again for the next step, the fused kernel can keep intermediate values closer to the compute units.

In PyTorch or JAX, compilation tools such as `torch.compile()` or `jax.jit()` can fuse operations when the graph is suitable.

The intuition is:

```
unfused: read -> op1 -> write -> read -> op2 -> write
fused:   read -> op1 -> op2 -> write
```

Fusion helps most when the bottleneck is memory movement or many small kernel launches.

## Recomputation

Training stores intermediate activations from the forward pass so the backward pass can compute gradients. For large models, those activations can become a major memory cost.

Recomputation trades compute for memory. Instead of saving every intermediate activation, the training loop saves fewer values and recomputes some forward activations during the backward pass.

This is useful when memory is the bottleneck:

- Save activations: faster backward pass, higher memory use.
- Recompute activations: lower memory use, extra compute.

The tradeoff is often worthwhile because GPU memory is a hard limit, while extra compute may be acceptable if it allows a larger batch, longer context, or larger model.

## Memory Coalescing and DRAM

DRAM reads in bursts. When one location is accessed, nearby data in the same burst section is delivered too.

If threads in a warp access nearby locations that fall within the same burst, the access is coalesced. This is efficient because the hardware can serve the warp with fewer memory transactions.

If threads access scattered locations, memory traffic becomes less efficient.

This matters for matrix layouts. If a computation repeatedly reads a full row or full column, it can be better to arrange or transpose the matrix so that the values read together are contiguous in memory. The math may be the same, but the memory access pattern changes.

## Tiling

Tiling groups computation into blocks so data can be reused from fast memory.

For matrix multiplication, a naive mental model is that each output element repeatedly reads individual values from global memory. Tiling instead loads sub-blocks of the input matrices into shared memory. Multiple threads then reuse those tiles to compute partial outputs before writing final results.

For example, instead of reading one number at a time from global memory, a kernel can load a small block, reuse it across several multiply-adds, then move to the next block.

`torch.matmul` handles this kind of optimization internally. In practice, the tile size depends on factors such as:

- coalesced memory access
- shared memory size
- matrix dimension divisibility
- streaming multiprocessor count

PyTorch also has compiler/autotuning options, such as max-autotune, that can search for better kernel choices or tiling configurations. If the tiling setup does not map well to the GPU's streaming multiprocessors, the work may require multiple passes.

## FlashAttention

Attention is a good example of how GPU-aware algorithms can change what gets stored and moved.

Naive attention materializes large intermediate matrices, including attention scores and softmax probabilities. For long sequences, those intermediates are expensive in memory.

FlashAttention reorganizes the computation around tiles:

- compute inner products tile by tile
- fuse operations such as exponentiation where possible
- compute softmax online using a numerically stable running-sum trick
- avoid storing huge attention matrices
- recompute pieces during the backward pass tile by tile

The core idea is not that attention math changes. The result is still attention. The implementation changes the memory behavior so much that the same math becomes much more practical on GPU hardware.

## Data Loading Can Starve the GPU

One practical training bug was data sampling. Using `np.random.choice(n, batch_size, replace=False)` looks harmless, but for a dataset with hundreds of millions of tokens it can force NumPy to scan or shuffle a huge population to guarantee no duplicates.

That makes data loading \(O(n)\) per step. The symptom is misleading: the model is on CUDA, but GPU utilization stays near zero because the CPU spends seconds preparing each batch.

The fix is independent random starts:

```python
indices = np.random.randint(0, n - context_length, batch_size)
```

This is \(O(\text{batch size})\). The chance of duplicate samples is negligible for large datasets:

\[
\frac{\text{batch size}^2}{2n}
\]

For a batch size of 32 and \(n = 200\text{M}\), the collision probability is roughly:

\[
\frac{32^2}{2 \cdot 200{,}000{,}000}
\approx 0.0000025\%
\]

In practice, the small possibility of repeated samples is much less important than keeping the accelerator fed.

## Implementation Checklist

When checking GPU utilization, I would ask:

- Is the model actually on the GPU?
- Are tensors staying on the GPU during the hot path?
- Is the CPU data loader fast enough to keep the GPU busy?
- Are kernels doing large, regular tensor operations instead of many tiny operations?
- Are branches causing warp divergence?
- Are masks or fused operations a better fit than explicit control flow?
- Are low-precision formats being used where they help?
- Are numerically sensitive operations kept in higher precision where needed?
- Is memory access coalesced?
- Can tiling, fusion, or recomputation reduce global memory traffic?
- Is an algorithm such as FlashAttention avoiding large attention intermediates?

The main intuition is that GPU efficiency is mostly about feeding the hardware the right shape of work. Dense tensor math is only part of the story; memory movement, branching, kernel fusion, data loading, and recomputation all affect whether the GPU is actually busy.
