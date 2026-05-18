# Supervised Learning Fundamentals

Supervised learning starts with paired examples: inputs and target outputs. The model learns a mapping from features to targets, and the kind of target determines the problem family.

## Classification and Regression

The two common supervised learning setups are:

- Classification: the target is a discrete class. The model tries to assign an input to one of several categories, such as spam versus not spam or benign versus malignant.
- Regression: the target is continuous. The model predicts a real-valued quantity, such as price, demand, or temperature.

The distinction matters because it changes both the output representation and the training objective:

- Classification models usually estimate class probabilities or decision scores.
- Regression models usually minimize a distance between predicted and true numerical values.

## Ranking, Multi-Class, and Multi-Label

Classification and ranking answer different kinds of questions:

- Classification predicts one or more labels for each item independently.
- Ranking predicts a relative ordering of items, usually to decide which items should appear higher for a given query, user, or context.

Multi-class and multi-label classification are also different:

- Multi-class classification: each example belongs to one class out of several possible classes.
- Multi-label classification: each example can receive multiple labels at the same time.

Ranking problems have different training setups:

- Pointwise: score each item independently.
- Pairwise: learn preferences between pairs of items.
- Listwise: optimize over the whole ranked list.

As the setup moves from pointwise to listwise, it models ranking structure more directly, but training becomes more complex.

## Prediction Unit

The prediction unit matters because the model should make predictions at the same granularity as the real decision. If the intervention happens at the user level but the model predicts at the event level, labels, features, and metrics may all become misaligned.

This can create a model that performs well offline but is not useful in production.

## Binary Classification and Anomaly Detection

Binary classification and anomaly detection can both produce a "normal vs unusual" decision, but they are framed differently:

- Binary classification: learn from examples of both classes and classify inputs into two categories.
- Anomaly detection: treat most data as coming from one normal distribution or pattern, then identify outliers from that pattern.

## Labels, Objectives, and Metrics

A true objective is the real outcome we ultimately care about. A proxy label is a measurable substitute used because the true objective is hard, delayed, or expensive to obtain.

For example, if the real objective is whether an app is popular, proxy labels might include:

- Number of users.
- Average time spent in the app.
- Retention.

The main risk is that optimizing the proxy may not fully optimize the true goal.

Offline metrics and business metrics can also mismatch:

- Offline metric: model performance on historical or benchmark data.
- Business metric: actual product or organizational outcome in the real world.

For example, an offline chatbot metric might measure whether answers are technically correct. A business metric might care about usage, retention, or whether users actually find the bot helpful.

## Bias, Variance, and Generalization

Bias and variance describe two different sources of error:

- Bias: error from assumptions that are too simple for the data-generating process. A high-bias model misses structure because its hypothesis class is too constrained.
- Variance: error from sensitivity to the particular training sample. A high-variance model may fit training data closely while failing on unseen examples.

Generalization is the central goal: the model should perform well on future data, not merely on the examples used for fitting. Most practical modeling choices are tradeoffs around this tension.

Underfitting and overfitting show up differently:

- Underfitting: high training error and high validation error. The model has not learned enough of the generic pattern.
- Overfitting: low training error and high validation error. The model fits training data too closely, including noise.
- Good fit: low training error and low validation error.

Common variance reduction methods:

- Add more representative training data.
- Reduce model complexity.
- Use stronger regularization.
- Stop training earlier.
- Use bagging.

Data size also matters. More representative and diverse data usually helps generalization. But more data that does not add new information, such as duplicated data, can make the model overconfident about a narrow pattern.

Model complexity is the other side of the tradeoff:

- More complex models can capture richer patterns, but they are harder to train and more prone to overfitting.
- Simpler models are easier to interpret and train, but they may underfit and have high bias.

Ensembles help in different ways:

- Bagging combines unstable learners and reduces variance.
- Boosting reduces bias by sequentially correcting mistakes.
- Stacking trains a meta-model that combines multiple models with complementary strengths.

## Train, Validation, and Test Splits

Each split has a different job:

- Training set: fit model parameters.
- Validation set: model selection, hyperparameter tuning, decision threshold selection, and early stopping.
- Test set: estimate performance on unseen data after training and model selection are done.

The key discipline is to avoid letting test-set feedback leak into the modeling process. Once the test set starts influencing choices, it stops being a clean estimate of future performance.

## Split Strategy

Common split strategies:

- Cross-validation: split the data into \(K\) folds. Each fold becomes the validation fold once, while the other \(K - 1\) folds are used for training. This is mainly useful when a single validation split feels too noisy.
- Stratified split: keep class proportions roughly the same across train, validation, and test. If 10% of the full dataset has positive labels, then each split should also have roughly 10% positive examples.
- Time-based split: train on earlier data and validate or test on later data, so future information does not leak backward.
- Group split: keep rows from the same entity in one split, such as multiple medical records from the same patient. Otherwise near-duplicate information can leak across train and test.

## Leakage

Two leakage patterns:

- Label leakage: features contain information that directly or indirectly reveals the target, especially information that would not be available at prediction time.
- Train-test contamination: the same or highly duplicated data appears in both the training set and validation or test set.

Leakage can also happen through preprocessing:

- Scaling or imputation should be fit on the training data after the split.
- The fitted preprocessing should then be applied to validation and test.
- If preprocessing statistics are computed using the full dataset first, validation and test information has already influenced training.

## Early Stopping and Tuning

Early stopping is a regularization method. It stops training when validation performance stops improving, so the model does not keep learning details that only help the training set.

A clean hyperparameter workflow:

1. Split the data into train, validation, and test.
2. Train models with different hyperparameters on the training set.
3. Compare them on the validation set.
4. Choose the best hyperparameters.
5. Test once on the test set.
