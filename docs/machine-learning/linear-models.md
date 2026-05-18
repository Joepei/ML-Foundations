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

Mean squared error is the average squared prediction error:

\[
\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2
\]

This objective penalizes large errors more than small ones, is always positive, and leads to a convex optimization problem. When the linear assumption is reasonable, the result is interpretable and efficient.

## Ordinary Least Squares

Ordinary least squares has a closed-form solution when the design matrix has full column rank. Setting the gradient of the residual sum of squares to zero gives the normal equations:

\[
\nabla_{\beta}\text{SSR}
= -2X^\top(y - X\beta) = 0
\]

\[
X^\top X \hat{\beta} = X^\top y
\]

So:

\[
\hat{\beta} = (X^\top X)^{-1}X^\top y
\]

Under the Gauss-Markov assumptions, OLS is BLUE: the best linear unbiased estimator. This is useful conceptually even when software uses more numerically stable methods than directly computing the inverse.

## Assumptions and Diagnostics

The usual linear regression assumptions matter most for inference about coefficients. If the goal is only prediction, violations can still hurt, but the interpretation changes.

Important assumptions include:

- Linearity: the conditional mean of \(y\) given \(x\) is linear. If this breaks, predictions are likely biased.
- Independent errors: the errors are uncorrelated. If this breaks, coefficient estimates can still be unbiased, but significance tests can become overconfident because the data contains less independent information than it appears to.
- Homoscedasticity: the variance of the errors is roughly constant around different values of \(x\). Intuitively, the "thickness" of the \(y\) values around the fitted line should be similar across the range of \(x\).
- Low multicollinearity: no feature is a near-linear combination of other features. If this breaks, small changes in \(X\) can lead to large changes in coefficients.
- Normality of errors for inference: mainly needed for exact small-sample inference, such as t-tests and confidence intervals.
- Exogeneity / zero conditional mean:

\[
E[\epsilon \mid X] = 0
\]

After controlling for \(X\), the remaining error should have no systematic relationship with \(X\). Equivalently, the model should not be leaving out a hidden factor that is related to both \(X\) and \(y\).

Diagnostic plots make these assumptions more concrete:

- Residual-versus-fitted plots check for bias patterns and changing variance.
- Q-Q plots check whether residuals are roughly normal.
- Scale-location plots show the square root of the absolute standardized residuals against fitted values. They are another way to look for changing variance.
- Residual-versus-leverage plots help identify influential points with unusual \(x\) values and large residuals.

Leverage comes from the diagonal of the hat matrix. A high-leverage point has an unusual \(x\) value, so its fitted value depends more heavily on its own training response.

Outliers hurt linear regression because squared error gives large residuals a large penalty. An unusual \(y\) value can dominate the loss, and an unusual \(x\) value can strongly pull the fitted line.

The worst case is an influential point with both unusual \(x\) and unusual \(y\), because it can substantially change the coefficient estimates.

## Feature Transformations

Linear models can represent more than straight-line relationships if the features are transformed first:

- Interaction terms: let the effect of one feature depend on another feature.
- Polynomial features: let a linear model capture nonlinear relationships while staying linear in the coefficients.
- Dummy variables: represent categorical variables with indicator features.

For a categorical variable with \(k\) possible values, we usually create \(k - 1\) dummy columns, not \(k\). Using all \(k\) creates perfect collinearity because one category can be inferred from the others.

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

Useful sigmoid properties:

| Property | Value |
| --- | --- |
| \(\sigma(0)\) | \(0.5\) |
| \(\sigma(\infty)\) | \(1\) |
| \(\sigma(-\infty)\) | \(0\) |
| \(\sigma(-z)\) | \(1 - \sigma(z)\) |
| \(\sigma'(z)\) | \(\sigma(z)(1 - \sigma(z))\) |

The odds are:

\[
\text{odds} = \frac{p}{1-p}
\]

The odds have an interpretable meaning: how likely class 1 is relative to class 0. Taking the log turns multiplicative effects on odds into additive effects:

\[
\log(A \cdot B \cdot C) = \log A + \log B + \log C
\]

This is why logistic regression can model log-odds with a linear expression. The log also maps odds from \((0, \infty)\) into \((-\infty, \infty)\), matching the range of a linear model.

## Log Loss and Thresholding

Logistic regression is usually trained with cross-entropy, or log loss. For binary labels:

We model each label \(y_i \in \{0,1\}\) as a Bernoulli random variable with probability:

\[
\hat{p}_i = \sigma(w^\top x_i + b)
\]

The likelihood of observing the data is:

\[
\mathcal{L}(w) = \prod_{i=1}^{n} \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}
\]

Taking the log makes it easier to optimize:

\[
\log \mathcal{L}(w)
= \sum_{i=1}^{n}
\left[
y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)
\right]
\]

Maximizing log likelihood is the same as minimizing negative log likelihood. Averaging gives the usual cross-entropy loss:

\[
-\frac{1}{n}\sum_i \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
\]

The model outputs probabilities, but classification decisions come from a threshold. With the usual threshold of 0.5, the decision boundary is:

\[
w^\top x + b = 0
\]

Changing the threshold changes the classification decision, but it does not change the underlying probability surface. This is why a model can improve in log loss but not necessarily improve a thresholded metric like accuracy, precision, or recall.

For example:

- Model A gives confident probabilities but makes one very confident mistake.
- Model B gives milder probabilities that classify examples correctly at the chosen threshold.

Depending on the objective, Model B can be better for classification even if Model A looks better in probability terms. If the task is risk scoring or ranking, the raw probabilities may matter more.

When the task involves thresholding, curves like ROC or precision-recall can help choose a threshold on the validation set.

Class imbalance changes how the model behaves. If one class is much more common, a model can lean heavily toward the majority class and still look decent on some metrics.

Common responses:

- Upweight the minority class in the loss.
- Oversample the minority class or undersample the majority class.
- Adjust the decision threshold to flag more minority-class examples.
- Use metrics such as precision, recall, F1, ROC-AUC, or PR-AUC instead of accuracy alone.

Calibration asks whether predicted probabilities match actual frequencies. If a model assigns probability near \(0.x\) to many examples, then roughly \(x\%\) of those examples should be positive for the probabilities to be well calibrated.

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
