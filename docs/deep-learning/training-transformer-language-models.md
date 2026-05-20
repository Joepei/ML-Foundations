# Training Transformer Language Models

Training a transformer language model is the part where the architecture becomes a working system. The loop has to coordinate the prediction objective, optimizer, learning-rate schedule, data loader, gradient handling, logging, validation, and checkpointing.

At a high level:

```
tokens -> batch -> forward pass -> loss -> backward pass -> optimizer step
```

The model learns by predicting the next token at every position in the sequence.

## Next-Token Prediction

For each position, the model receives the prefix up to that position and predicts the next token. The output of the transformer is a tensor of logits:

\[
(\text{batch}, \text{sequence}, \text{vocab size})
\]

The targets have shape:

\[
(\text{batch}, \text{sequence})
\]

Each target entry is the true next-token ID for that position.

The loss is cross entropy between the model's predicted distribution and the true next token:

\[
\mathcal{L} = -\log p(y)
\]

In implementation, it is better to compute cross entropy from logits directly instead of explicitly forming probabilities first. The stable version subtracts the maximum logit before exponentiating:

```python
max_values = predicted_logits.max(dim=-1, keepdim=True).values
predicted_logits = predicted_logits - max_values
target_logits = predicted_logits.gather(dim=-1, index=targets.unsqueeze(-1))
loss = -(target_logits - torch.log(predicted_logits.exp().sum(dim=-1, keepdim=True))).mean()
```

Two details matter here:

- `gather` needs `targets.unsqueeze(-1)` so the index tensor has the same rank as the logits.
- Subtracting the max logit improves numerical stability without changing the softmax probabilities.

## Data Loading

The tokenized dataset is treated as one long sequence:

\[
x = (x_1, x_2, \ldots, x_n)
\]

Even if the original text came from separate documents, it is common to concatenate the documents with a delimiter token such as `<|endoftext|>`.

A data loader samples subsequences of length \(m\), then pairs each sequence with the same span shifted by one token. For \(B = 1\) and \(m = 3\), one batch element could be:

\[
([x_2, x_3, x_4], [x_3, x_4, x_5])
\]

The implementation samples random starting positions, then uses broadcasting to build rolling indices:

```python
indices = np.random.randint(0, n - context_length, batch_size)
rolling_indices = np.array(indices)[:, None] + np.arange(context_length)

inputs = torch.from_numpy(x[rolling_indices]).to(device)
targets = torch.from_numpy(x[rolling_indices + 1]).to(device)
```

Using `np.random.randint` is important. A tempting alternative is `np.random.choice(..., replace=False)`, but that can scan or shuffle a huge population to guarantee no duplicates. For a dataset with hundreds of millions of tokens, that turns data loading into a CPU bottleneck and leaves the GPU idle.

## AdamW

AdamW combines Adam's adaptive updates with decoupled weight decay.

Adam starts from the momentum idea: instead of using only the current gradient, track an exponentially weighted average of past gradients:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]

Momentum smooths noisy gradients, but it does not solve the problem that parameters with consistently large gradients can keep taking large steps. Large gradient magnitude does not necessarily mean the update is better for reducing the loss.

Adam adds a second moment estimate:

\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\]

The update is scaled by the square root of this gradient-history estimate, so parameters with large historical gradients get damped and parameters with smaller historical gradients can still move.

The implementation also applies bias correction through the effective learning rate:

```python
lr = group["lr"] * ((1 - beta2**step) ** 0.5) / (1 - beta1**step)
```

The key distinction in AdamW is weight decay. If weight decay is mixed directly into Adam's gradient update, the penalty becomes entangled with Adam's adaptive machinery. AdamW decouples it by applying weight decay directly to the parameter:

```python
theta.data -= group["lr"] * group["weight_decay"] * theta.data
theta.data -= lr * m / (v ** 0.5 + group["eps"])
```

That makes the regularization easier to reason about and tune.

## Learning-Rate Schedule

The training loop updates the learning rate every step:

```python
lr = learning_rate_schedule(step, lr_max, lr_min, warmup_steps, total_steps)
for group in optimizer.param_groups:
    group["lr"] = lr
```

The schedule has two phases:

1. Warm up linearly from 0 to `lr_max`.
2. Decay from `lr_max` toward `lr_min` with a cosine schedule.

The implementation:

```python
def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        return t / Tw * alpha_max
    elif Tw <= t <= Tc:
        return alpha_min + 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min
```

Warmup is useful because the optimizer's moving averages are still settling early in training. The cosine decay then gradually reduces the step size as training progresses.

## Gradient Clipping

Some batches can produce very large gradients. If the optimizer takes a step with those gradients directly, training can become unstable.

Gradient clipping limits the norm of the full gradient vector after backpropagation and before the optimizer step:

```python
loss.backward()
gradient_clipping(model.parameters(), grad_clip)
optimizer.step()
```

The clipping is global across all parameters:

\[
\|\nabla\| =
\sqrt{\sum_i \|\nabla_{\theta_i}\|^2}
\]

If the norm exceeds the maximum, each gradient is scaled by the same factor:

```python
theta.grad *= maximum / (norm + eps)
```

This preserves the direction of the full gradient vector while limiting its magnitude.

## Checkpointing

Checkpointing saves enough state to resume training:

```python
torch.save(
    {
        "Model": model.state_dict(),
        "Optimizer": optimizer.state_dict(),
        "Iteration": iteration,
    },
    out,
)
```

The model state is not enough by itself. The optimizer state matters because AdamW tracks moving averages and step counts. Resuming without optimizer state changes the training dynamics.

Loading restores both:

```python
d = torch.load(src, map_location="cpu")
model.load_state_dict(d["Model"])
optimizer.load_state_dict(d["Optimizer"])
start_step = d["Iteration"]
```

## Training Loop

The training loop has a fixed rhythm:

1. Load tokenized train and validation data.
2. Initialize the model and optimizer.
3. Optionally resume from a checkpoint.
4. For each step, compute the current learning rate.
5. Sample a batch.
6. Run the forward pass.
7. Compute cross entropy loss.
8. Clear old gradients with `optimizer.zero_grad()`.
9. Run `loss.backward()`.
10. Clip gradients.
11. Step the optimizer.
12. Log training loss.
13. Periodically evaluate on validation batches.
14. Periodically save checkpoints.

The core update sequence is:

```python
logits = model(inputs)
loss = cross_entropy(logits, targets)

optimizer.zero_grad()
loss.backward()
gradient_clipping(model.parameters(), args.grad_clip)
optimizer.step()
```

The order matters. Old gradients must be cleared before backpropagation, clipping must happen after gradients exist, and the optimizer step should happen after clipping.

## Validation and Logging

Training loss is useful, but validation loss tells whether the model is improving on held-out data.

The implementation periodically switches to evaluation mode and disables gradient tracking:

```python
model.eval()
with torch.no_grad():
    val_losses = []
    for _ in range(20):
        x, y = data_loading(val_data, batch_size, context_length, device)
        val_losses.append(cross_entropy(model(x.long()), y.long()).item())
val_loss = sum(val_losses) / len(val_losses)
model.train()
```

The loop also supports CSV logging:

```python
log_writer.writerow(["step", "wall_time", "train_loss", "val_loss"])
```

This can later become a plot of training loss, validation loss, and wall-clock time.

## Sanity Checks

Before running full training, overfit a single minibatch.

The implementation supports this by sampling one fixed batch at startup and reusing it every step:

```python
if args.overfit_batch:
    fixed_inputs, fixed_targets = data_loading(...)

for step in range(...):
    if args.overfit_batch:
        inputs, targets = fixed_inputs, fixed_targets
```

The interpretation:

- Loss drops near 0: forward pass, backward pass, and optimizer are probably wired correctly.
- Loss does not move: something basic is wrong in the model, loss, or gradients.
- Loss decreases but plateaus too high: the learning rate may be too low or the model may be constrained.
- Loss explodes: the learning rate may be too high or gradients may be unstable.

Another practical note: step 0 is often slow because CUDA kernels compile or initialize on first use. A slow first step is not automatically a training bug.

## Implementation Checklist

When implementing a transformer training loop, I would check:

- Are inputs and targets shifted by exactly one token?
- Are logits shaped `(batch, sequence, vocab_size)` and targets shaped `(batch, sequence)`?
- Is cross entropy computed from logits in a numerically stable way?
- Is the data loader \(O(\text{batch size})\), not accidentally \(O(n)\)?
- Is the learning rate updated before the optimizer step?
- Does AdamW keep first and second moment state per parameter?
- Is weight decay decoupled from the adaptive gradient update?
- Are gradients zeroed before `loss.backward()`?
- Is global gradient clipping applied before `optimizer.step()`?
- Are validation runs using `model.eval()` and `torch.no_grad()`?
- Do checkpoints include model state, optimizer state, and iteration?
- Can the model overfit one small batch before full training?

The main intuition is that training is not just "call backward." It is a coordinated system. A correct architecture can still fail if the data loader is slow, the loss is unstable, gradients are unclipped, optimizer state is not restored, or the loop is wired in the wrong order.
