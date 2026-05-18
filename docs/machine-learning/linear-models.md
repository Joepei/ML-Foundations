# Linear Models

Linear models are simple, but they remain useful because their assumptions are explicit and their behavior is easy to inspect. They also provide a foundation for understanding optimization, regularization, and probabilistic modeling.

## Linear Regression

Linear regression predicts a continuous target as a weighted combination of input features:

\[
\hat{y} = w^\top x + b
\]

The standard objective minimizes squared residuals:

\[
\sum_i (y_i - \hat{y}_i)^2
\]

This objective penalizes large errors strongly and leads to a convex optimization problem. When the linear assumption is reasonable, the result is interpretable and efficient.

## Logistic Regression

Logistic regression is a classification model. Instead of predicting a continuous value directly, it models the log-odds of a class as a linear function of the features:

\[
\log \frac{p(y = 1 \mid x)}{1 - p(y = 1 \mid x)} = w^\top x + b
\]

Equivalently, it passes the linear score through the sigmoid function:

\[
p(y = 1 \mid x) = \sigma(w^\top x + b)
\]

One useful interpretation is multiplicative odds. If a feature increases the log-odds by a fixed amount, it multiplies the odds by a fixed factor.

## Regularization

Regularization constrains model complexity so the fitted model is less sensitive to noise.

L2 regularization penalizes large weights:

\[
\lambda \lVert w \rVert_2^2
\]

L1 regularization penalizes absolute weight values:

\[
\lambda \lVert w \rVert_1
\]

L2 tends to shrink weights smoothly, while L1 can drive some weights to zero and produce sparse models.
