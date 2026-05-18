# Tree-Based Models

Tree-based models split feature space into regions and make predictions within each region. They are popular because they handle nonlinearity, feature interactions, and mixed feature types with relatively little preprocessing.

## Decision Trees

A decision tree learns a sequence of feature-based splits. Each split is chosen to reduce impurity in the resulting child nodes.

For classification, common impurity measures include Gini impurity and entropy. For regression, splits often minimize squared error within each region.

The impurity reduction for a split can be written as:

\[
\text{IG}(I) =
I(D_t) -
\left[
\frac{|D_L|}{|D_t|} I(D_L)
+ \frac{|D_R|}{|D_t|} I(D_R)
\right]
\]

Here \(D_t\) is the parent node data, and \(D_L\), \(D_R\) are the left and right child node data.

Trees are flexible, but a single deep tree can overfit. It may learn idiosyncratic rules that describe the training set better than the underlying pattern.

Trees also do not usually need feature scaling. The split only depends on the ordering of feature values and the impurity reduction, not the units of the feature.

## Random Forests

Random forests reduce variance by training many decision trees on randomized versions of the data and features.

The core idea is bagging: train each tree on a bootstrap sample, then average predictions for regression or take a majority vote for classification.

Random feature selection decorrelates the trees. If each tree makes somewhat different errors, averaging produces a more stable model.

## Out-of-Bag Validation

Each bootstrap sample contains roughly 63.2% unique observations from the original dataset. The remaining observations are out-of-bag for that tree.

Out-of-bag validation uses those held-out observations as a built-in validation signal. For each training point, aggregate predictions from the trees where that point was out-of-bag, then compute a validation score over the dataset.

## Feature Importance Caveats

Impurity-based feature importance can be biased. Continuous features, or categorical features with many possible split points, get more chances to reduce impurity and may look more important than they really are.

Permutation importance is one way to check this. Randomly shuffle one feature and measure how much model performance drops. If performance drops a lot, the model was relying on that feature.

## Gradient Boosting

Boosting trains models sequentially. Each new tree focuses on correcting the errors left by the previous ensemble.

Instead of building many independent trees, gradient boosting builds an additive model:

\[
F_m(x) = F_{m-1}(x) + \eta h_m(x)
\]

Here \(h_m\) is the next weak learner and \(\eta\) is the learning rate. The model improves by taking small steps in the direction that reduces the training objective.

For squared error loss, this correction is closely related to fitting residuals. For more general losses, the tree is fit to the negative gradient of the loss.

Bagging builds many independent trees and averages them. Boosting builds trees sequentially, so later trees depend on what earlier trees missed.
