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

## Ordinary Least Squares

Ordinary least squares has a closed-form solution when the design matrix has full column rank. Setting the gradient of the residual sum of squares to zero gives the normal equations:

\[
X^\top X \hat{\beta} = X^\top y
\]

So:

\[
\hat{\beta} = (X^\top X)^{-1}X^\top y
\]

This is useful conceptually even when software uses more numerically stable methods than directly computing the inverse.

## Assumptions and Diagnostics

The usual linear regression assumptions matter most for inference about coefficients. If the goal is only prediction, violations can still hurt, but the interpretation changes.

Important assumptions include:

- Independent errors.
- Roughly constant error variance.
- Low multicollinearity.
- Normality of errors for small-sample inference.
- Exogeneity:

\[
E[\epsilon \mid X] = 0
\]

Diagnostic plots make these assumptions more concrete:

- Residual-versus-fitted plots check for bias patterns and changing variance.
- Q-Q plots check whether residuals are roughly normal.
- Residual-versus-leverage plots help identify influential points with unusual \(x\) values and large residuals.

Outliers hurt linear regression because squared error gives large residuals a large penalty. An unusual \(y\) value can dominate the loss, and an unusual \(x\) value can strongly pull the fitted line.

The worst case is an influential point with both unusual \(x\) and unusual \(y\), because it can substantially change the coefficient estimates.

## Feature Transformations

Linear models can represent more than straight-line relationships if the features are transformed first:

- Interaction terms: let the effect of one feature depend on another feature.
- Polynomial features: let a linear model capture nonlinear relationships while staying linear in the coefficients.
- Dummy variables: represent categorical variables with indicator features.

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

The sigmoid function maps any real-valued score into \((0, 1)\):

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

## Log Loss and Thresholding

Logistic regression is usually trained with cross-entropy, or log loss. For binary labels:

\[
-\sum_i \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
\]

The model outputs probabilities, but classification decisions come from a threshold. With the usual threshold of 0.5, the decision boundary is:

\[
w^\top x + b = 0
\]

Changing the threshold changes the classification decision, but it does not change the underlying probability surface. This is why a model can improve in log loss but not necessarily improve a thresholded metric like accuracy, precision, or recall.

Class imbalance changes how the model behaves. If one class is much more common, a model can lean heavily toward the majority class and still look decent on some metrics.

Common responses:

- Upweight the minority class in the loss.
- Resample the data.
- Tune the decision threshold on the validation set.
- Use metrics such as precision, recall, F1, ROC-AUC, or PR-AUC instead of accuracy alone.

Calibration is about whether predicted probabilities match actual frequencies. If a model assigns probability near 0.7 to many examples, then roughly 70% of those examples should be positive for the probabilities to be well calibrated.

## Regularization

Regularization constrains model complexity so the fitted model is less sensitive to noise.

Two common penalties:

- L2 regularization penalizes large weights:

\[
\lambda \lVert w \rVert_2^2
\]

- L1 regularization penalizes absolute weight values:

\[
\lambda \lVert w \rVert_1
\]

L2 tends to shrink weights smoothly, while L1 can drive some weights to zero and produce sparse models.

Scaling does not change ordinary least squares predictions or \(R^2\), but it matters for regularization and for gradient-based optimization. If features are on very different scales, the penalty and optimization path can behave unevenly.
