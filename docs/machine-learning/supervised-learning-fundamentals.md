# Supervised Learning Fundamentals

Supervised learning starts with paired examples: inputs and target outputs. The model learns a mapping from features to targets, and the kind of target determines the problem family.

## Classification and Regression

In classification, the target is a discrete class. The model tries to assign an input to one of several categories, such as spam versus not spam or benign versus malignant.

In regression, the target is continuous. The model predicts a real-valued quantity, such as price, demand, or temperature.

The distinction matters because it changes both the output representation and the training objective. Classification models usually estimate class probabilities or decision scores; regression models usually minimize a distance between predicted and true numerical values.

## Bias, Variance, and Generalization

Bias is error from assumptions that are too simple for the data-generating process. A high-bias model misses structure because its hypothesis class is too constrained.

Variance is error from sensitivity to the particular training sample. A high-variance model may fit training data closely while failing on unseen examples.

Generalization is the central goal: the model should perform well on future data, not merely on the examples used for fitting. Most practical modeling choices are tradeoffs around this tension.

## Train, Validation, and Test Splits

The training set is used to fit model parameters. The validation set is used for model selection, hyperparameter tuning, and early stopping. The test set is held out until the end to estimate performance on unseen data.

The key discipline is to avoid letting test-set feedback leak into the modeling process. Once the test set starts influencing choices, it stops being a clean estimate of future performance.
