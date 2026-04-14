
# PCA Homework Answer Key

## Problem 1. Conceptual foundations of PCA

### Variance maximization view
1. Maximizing variance is useful because projections with larger variance preserve more of the structure and spread of the data.
2. PC1 is the direction along which the projected data have the largest possible variance.
3. PC2 must be orthogonal to PC1 so that it captures new variation not already represented by PC1.

### Decorrelation view
4. Two variables are uncorrelated if their covariance is zero.
5. Diagonalizing the covariance matrix makes all off-diagonal covariances zero in the new coordinate system.
6. PCA produces uncorrelated principal components because it projects data onto orthogonal eigenvectors of the covariance matrix.

### Covariance
For centered data matrix X, the covariance matrix is
Sigma = (1/(n-1)) X^T X.
Covariance is central to PCA because PCA seeks directions that explain variation and covariance measures joint variation among features.

## Problem 2. PCA derivation via eigenvectors
The variance of the projected data along unit vector w is w^T Sigma w.
PCA solves:
maximize w^T Sigma w subject to ||w|| = 1.
Using a Lagrange multiplier leads to:
Sigma w = lambda w.
Thus:
- eigenvectors = principal directions
- eigenvalues = variances explained by those directions
The eigenvector with the largest eigenvalue is PC1 because it gives the maximum projected variance.

## Problem 3. Manual PCA on the toy dataset
Data:
(2,2), (3,3), (4,4), (5,5)

1. Mean = (3.5, 3.5)
2. Centered data:
(-1.5,-1.5), (-0.5,-0.5), (0.5,0.5), (1.5,1.5)
3. Covariance matrix:
[[5/3, 5/3],
 [5/3, 5/3]]
4. Eigenvalues:
lambda1 = 10/3 ≈ 3.3333
lambda2 = 0
5. PC1 direction is proportional to (1,1), normalized to (1/sqrt(2), 1/sqrt(2))
6. There is essentially one principal component because all points lie exactly on the line x1 = x2.

## Problem 4. PCA implementation
1. Load the breast cancer dataset.
2. Standardize features before PCA.
3. Fit PCA and inspect explained variance ratio.
4. Plot cumulative explained variance.
5. The number of PCs needed for about 90% variance is determined from the cumulative explained variance plot.

## Problem 5. PCA visualization
1. Fit PCA with n_components=2.
2. Plot PC1 vs PC2 colored by class.
3. The classes often show noticeable separation, suggesting that disease status aligns with some dominant variation in the measurements.
4. PCA can reveal structure without labels because biologically meaningful variance may already organize the samples.

## Problem 6. PCA loadings
1. Loadings are entries of pca.components_.
2. A large absolute loading means the original feature strongly contributes to that principal component.
3. High-loading features may reflect important biological variation.
4. PCA can help exploratory biomarker discovery, but it is not sufficient by itself because it ignores labels and focuses on variance rather than prediction.

## Problem 7. PCA + classification pipeline
1. Compare logistic regression without PCA to logistic regression with PCA.
2. Evaluate accuracy and ROC AUC.
3. PCA may help by removing noise and redundancy.
4. PCA may hurt if predictive information lies in lower-variance directions or if too few PCs are retained.

## Problem 8. PCA vs original feature space
1. PCA reduces dimensionality but may discard information.
2. It can reduce noise and redundancy.
3. It may reduce overfitting in high-dimensional settings.
4. PCA is especially useful in omics because omics data often contain many correlated variables and require dimensionality reduction for visualization and modeling.

## Problem 9. Limitations of PCA
1. PCA is linear, so it may miss nonlinear biological patterns.
2. PCA ignores class labels, so high-variance directions may not be the best class-separating directions.
3. PCA maximizes variance, not biological importance.
4. PCA may fail when technical batch effects dominate or when subtle disease patterns lie in nonlinear or low-variance directions.
