# Statistical Inference and Significance Tests

Statistical inference asks how much we should trust a pattern estimated from data. In machine learning, this shows up when we want to know whether a model improvement, coefficient, or observed difference is likely to reflect a real signal rather than sampling noise.

The important shift is from "what did I observe?" to "how surprising would this observation be under a baseline assumption?"

## Hypothesis Tests

A significance test starts with two hypotheses:

- Null hypothesis \(H_0\): the baseline claim, usually that there is no effect or no difference.
- Alternative hypothesis \(H_1\): the claim that there is an effect, difference, or association.

Examples:

- A coefficient test might use \(H_0: \beta_j = 0\), meaning the feature has no association with the target after controlling for the other features.
- A model comparison test might use \(H_0\): two models have the same expected performance.
- An experiment might use \(H_0\): treatment and control have the same conversion rate.

The test statistic compresses the observed data into a number that measures how far the result is from what the null hypothesis predicts.

## P-Values

A p-value is the probability of seeing a result at least as extreme as the observed result, assuming the null hypothesis is true.

The phrase "assuming the null is true" is the part that matters. A p-value is not the probability that the null hypothesis is true. It is a statement about how surprising the observed data would be under the null.

Small p-values mean the observation would be unusual if the null were true. They do not automatically mean the effect is large, important, causal, or useful in production.

Common failure modes:

- Treating \(p < 0.05\) as proof instead of evidence.
- Ignoring effect size.
- Running many tests and only reporting the significant ones.
- Reusing the test set until it becomes part of the modeling process.
- Forgetting that dependence between samples can make a dataset look more informative than it really is.

## T-Statistics

A t-statistic compares an estimated effect to its uncertainty:

\[
t = \frac{\text{estimate} - \text{null value}}{SE(\text{estimate})}
\]

The mental model is simple: an effect is more convincing when it is large relative to its standard error.

For a regression coefficient, the null value is often zero:

\[
t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
\]

If the standard error is too small, the t-statistic becomes too large. This is why assumptions about independence and variance matter. The numerator is the estimated effect; the denominator is our estimate of uncertainty.

## Confidence Intervals

A confidence interval gives a range of plausible values for the parameter:

\[
\text{estimate} \pm \text{critical value} \cdot SE(\text{estimate})
\]

For a regression coefficient, this often looks like:

\[
\hat{\beta}_j \pm t_{\alpha/2, df} \cdot SE(\hat{\beta}_j)
\]

Confidence intervals and significance tests are closely related. If a 95% confidence interval for a coefficient does not include zero, the corresponding two-sided test is usually significant at the 5% level.

The interval is often more useful than the p-value because it shows both direction and scale. A tiny p-value with a tiny effect may be statistically detectable but practically unimportant.

## Independence and Effective Sample Size

Significance tests depend heavily on how much independent information the data contains.

If observations are correlated, the nominal row count can be misleading. For example, 100 rows from 5 highly correlated groups may behave more like 5 independent pieces of information than 100. The estimate itself may still point in the right direction, but the uncertainty can be underestimated.

That creates overconfident inference:

- Standard errors look too small.
- Test statistics look too large.
- P-values look too impressive.
- Confidence intervals look too narrow.

This is why grouped data, repeated measurements, time series, and duplicated examples need extra care.

## Prediction Versus Inference

Prediction and inference care about overlapping but different questions.

For prediction:

- The main question is whether the model generalizes to future data.
- Validation and test performance matter more than whether every coefficient is significant.
- Assumption violations matter when they hurt generalization or make evaluation misleading.

For inference:

- The main question is whether the estimated effect is trustworthy.
- Standard errors, p-values, and confidence intervals matter.
- Assumptions about independence, variance, omitted variables, and sampling are central.

This is the clean separation: prediction asks whether the model works; inference asks how much we trust a specific estimated effect.
