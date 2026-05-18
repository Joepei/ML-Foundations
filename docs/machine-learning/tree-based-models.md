# Tree-Based Models

Tree-based models split feature space into regions and make predictions within each region. They are popular because they handle nonlinearity, feature interactions, and mixed feature types with relatively little preprocessing.

## Decision Trees

A decision tree learns a sequence of feature-based splits. Each split is chosen to reduce impurity in the resulting child nodes.

The splitting process is greedy. At each node, the tree chooses the locally best split according to impurity reduction, without knowing whether that split will lead to the globally best full tree.

Common impurity measures:

- Classification: Gini impurity or entropy.
- Regression: squared error within each region.

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

For regression trees, the prediction at each leaf is usually the mean target value of the training samples in that leaf. A split helps when it reduces the target variance inside the resulting leaves.

Trees also do not usually need feature scaling. The split only depends on the ordering of feature values and the impurity reduction, not the units of the feature.

Stopping criteria control tree complexity:

- Max depth: stop when the tree reaches a certain depth.
- Min samples split: split only if a node has enough samples.
- Min samples leaf: require each leaf to have enough samples.
- Min impurity decrease: split only if impurity reduction is large enough.
- Max leaves: limit the total number of leaf nodes.

Pruning is another way to control complexity. Instead of only stopping early, grow a larger tree and remove branches that do not improve validation performance enough.

Trees capture nonlinear effects because they partition feature space into regions. They capture interactions because later splits depend on earlier splits.

## Random Forests

Random forests reduce variance by training many decision trees on randomized versions of the data and features.

The core idea is bagging:

- Train each tree on a bootstrap sample.
- Average predictions for regression.
- Take a majority vote for classification.

Random feature selection decorrelates the trees. If each tree makes somewhat different errors, averaging produces a more stable model.

Random forests add two layers of randomness:

- Bootstrap sampling: each tree sees a different sampled dataset.
- Feature subsampling: at each split, the tree only considers a random subset of features.

This is why random forests are more stable than single trees. They average many plausible versions of the training process, so predictions become smoother.

## Out-of-Bag Validation

Each bootstrap sample contains:

- Roughly 63.2% unique observations from the original dataset.
- Roughly 36.8% out-of-bag observations for that tree.

Out-of-bag validation uses those held-out observations as a built-in validation signal:

- For each training point, collect predictions from the trees where that point was out-of-bag.
- Aggregate those predictions.
- Compute a validation score over the dataset.

## Feature Importance Caveats

Impurity-based feature importance can be biased:

- Continuous features get many possible split points.
- Categorical features with many values also get many possible split points.
- More split opportunities can make a feature look more important than it really is.

Permutation importance is one way to check this:

- Randomly shuffle one feature.
- Measure how much model performance drops.
- If performance drops a lot, the model was relying on that feature.

## Gradient Boosting

Boosting trains models sequentially. Each new tree focuses on correcting the errors left by the previous ensemble.

Instead of building many independent trees, gradient boosting builds an additive model:

\[
F_m(x) = F_{m-1}(x) + \eta h_m(x)
\]

Here \(h_m\) is the next weak learner and \(\eta\) is the learning rate. The model improves by taking small steps in the direction that reduces the training objective.

The correction depends on the loss:

- For squared error loss, this is closely related to fitting residuals.
- For more general losses, the tree is fit to the negative gradient of the loss.

Bagging and boosting differ in how trees are built:

- Bagging builds many independent trees and averages them.
- Boosting builds trees sequentially, so later trees depend on what earlier trees missed.

Important tuning knobs:

- Learning rate and number of trees are usually tuned together. Smaller learning rates often need more trees.
- Tree depth controls how complex each correction can be. Boosting often uses shallow trees because each tree is meant to be a small correction.
- Overfitting shows up when training loss keeps improving but validation loss stops improving.

At a high level:

- XGBoost: regularized gradient boosting with strong support for sparse data, missing values, row/column sampling, and second-order optimization.
- LightGBM: histogram-based boosting designed for speed and scale, especially on large tabular datasets.
- CatBoost: boosting designed to handle categorical features well, using ordered target statistics to reduce target leakage.
