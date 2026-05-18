# Training Transformer Language Models

Training a transformer language model combines a prediction objective, an optimizer, a data loader, and a loop that coordinates forward passes, gradients, clipping, logging, validation, and checkpointing.

## Next-Token Prediction

For each token position, the model predicts the next token. The output is a probability distribution over the vocabulary, and the target is the actual next token.

The standard loss is cross entropy:

\[
\mathcal{L} = - \sum_i y_i \log \hat{p}_i
\]

For next-token prediction, this is applied across positions and batches.

## AdamW

AdamW combines Adam's adaptive update rule with decoupled weight decay.

Momentum uses an exponentially weighted moving average of past gradients. Adam extends this idea by also tracking a second moment estimate, which scales updates according to each parameter's gradient history.

Weight decay should not be entangled with Adam's adaptive machinery. AdamW decouples it by applying weight decay directly to the parameters, which makes the regularization easier to tune.

## Gradient Clipping

Occasionally, a batch can produce very large gradients. Gradient clipping limits the norm of the full gradient vector before the optimizer step.

This is applied globally across parameters, then individual gradients are scaled proportionally if the norm exceeds the threshold.

## Data Loading

Tokenized training data is often treated as one long sequence of token IDs. A data loader samples subsequences of length \(m\), then pairs each sequence with the same sequence shifted by one token:

\[
([x_2, x_3, x_4], [x_3, x_4, x_5])
\]

This creates input-target pairs for next-token prediction.

## Training Loop

A minimal training loop has the following rhythm:

1. Load tokenized data.
2. Initialize the model and optimizer.
3. Sample a batch.
4. Run the forward pass.
5. Compute the loss.
6. Clear old gradients.
7. Run backpropagation.
8. Clip gradients.
9. Step the optimizer.
10. Log metrics, validate, and checkpoint.

A useful sanity check is to overfit a tiny minibatch before training on the full dataset. If the model cannot drive loss near zero on a tiny batch, something basic is likely wrong.
