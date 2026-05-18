# Tree-Based Models

Tree-based models split feature space into regions and make predictions within each region. They are popular because they handle nonlinearity, feature interactions, and mixed feature types with relatively little preprocessing.

## Decision Trees

A decision tree learns a sequence of feature-based splits. Each split is chosen to reduce impurity in the resulting child nodes.

For classification, common impurity measures include Gini impurity and entropy. For regression, splits often minimize squared error within each region.

Trees are flexible, but a single deep tree can overfit. It may learn idiosyncratic rules that describe the training set better than the underlying pattern.

## Random Forests

Random forests reduce variance by training many decision trees on randomized versions of the data and features.

The core idea is bagging: train each tree on a bootstrap sample, then average predictions for regression or take a majority vote for classification.

Random feature selection decorrelates the trees. If each tree makes somewhat different errors, averaging produces a more stable model.

## Gradient Boosting

Boosting trains models sequentially. Each new tree focuses on correcting the errors left by the previous ensemble.

Instead of building many independent trees, gradient boosting builds an additive model:

\[
F_m(x) = F_{m-1}(x) + \eta h_m(x)
\]

Here \(h_m\) is the next weak learner and \(\eta\) is the learning rate. The model improves by taking small steps in the direction that reduces the training objective.
