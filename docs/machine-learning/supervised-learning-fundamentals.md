# Supervised Learning Fundamentals

Supervised learning starts with paired examples: inputs and target outputs. The model learns a mapping from features to targets, and the kind of target determines the problem family.

## Classification and Regression

In classification, the target is a discrete class. The model tries to assign an input to one of several categories, such as spam versus not spam or benign versus malignant.

In regression, the target is continuous. The model predicts a real-valued quantity, such as price, demand, or temperature.

The distinction matters because it changes both the output representation and the training objective. Classification models usually estimate class probabilities or decision scores; regression models usually minimize a distance between predicted and true numerical values.

## Ranking, Multi-Class, and Multi-Label

Classification predicts one or more labels for each item independently. Ranking predicts a relative ordering of items, usually to decide which items should appear higher for a given query, user, or context.

In multi-class classification, each example belongs to one class out of several possible classes. In multi-label classification, each example can receive multiple labels at the same time.

Ranking problems also have different training setups. A pointwise method scores each item independently. A pairwise method learns preferences between pairs of items. A listwise method optimizes over the whole ranked list. As the setup moves from pointwise to listwise, it models ranking structure more directly, but training becomes more complex.

## Prediction Unit

The prediction unit matters because the model should make predictions at the same granularity as the real decision. If the intervention happens at the user level but the model predicts at the event level, labels, features, and metrics may all become misaligned.

This can create a model that performs well offline but is not useful in production.

## Bias, Variance, and Generalization

Bias is error from assumptions that are too simple for the data-generating process. A high-bias model misses structure because its hypothesis class is too constrained.

Variance is error from sensitivity to the particular training sample. A high-variance model may fit training data closely while failing on unseen examples.

Generalization is the central goal: the model should perform well on future data, not merely on the examples used for fitting. Most practical modeling choices are tradeoffs around this tension.

## Train, Validation, and Test Splits

The training set is used to fit model parameters. The validation set is used for model selection, hyperparameter tuning, and early stopping. The test set is held out until the end to estimate performance on unseen data.

The key discipline is to avoid letting test-set feedback leak into the modeling process. Once the test set starts influencing choices, it stops being a clean estimate of future performance.

## Split Strategy

Cross-validation splits the data into \(K\) folds. Each fold becomes the validation fold once, while the other \(K - 1\) folds are used for training. This is mainly useful for model selection and hyperparameter tuning when a single validation split feels too noisy.

A stratified split keeps class proportions roughly the same across train, validation, and test. If 10% of the full dataset has positive labels, then each split should also have roughly 10% positive examples.

A time-based split is used when data has temporal order. The model should train on earlier data and validate or test on later data, so future information does not leak backward.

A group split is useful when multiple rows belong to the same entity, such as multiple medical records from the same patient. The group should stay in one split, otherwise near-duplicate information can leak across train and test.

## Leakage

Label leakage happens when features contain information that directly or indirectly reveals the target, especially information that would not be available at prediction time.

Train-test contamination is different: it happens when the same or highly duplicated data appears in both the training set and validation or test set.

Leakage can also happen through preprocessing. For example, scaling or imputation should be fit on the training data after the split, then applied to validation and test. If preprocessing statistics are computed using the full dataset first, validation and test information has already influenced training.
