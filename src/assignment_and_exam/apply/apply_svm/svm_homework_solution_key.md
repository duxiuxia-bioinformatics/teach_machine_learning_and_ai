
# Support Vector Machines for Bioinformatics Classification
## Full Solution Key

### Part I — Conceptual Foundations

**Q1. Hyperplane and margin**
- A hyperplane is a flat decision boundary in p-dimensional feature space.
- The margin is the perpendicular distance from the decision boundary to the closest training points.
- The distance from the decision boundary \(w^T x + b = 0\) to the margin boundary \(w^T x + b = 1\) is \(1/\|w\|\). The full margin width is \(2/\|w\|\).

**Q2. Hinge loss**
Using \(L(y,f(x)) = \max(0, 1 - yf(x))\):

| Sample | y | f(x) | y f(x) | Hinge loss | Interpretation |
|---|---:|---:|---:|---:|---|
| 1 | +1 | 1.8 | 1.8 | 0.0 | correctly classified outside margin |
| 2 | +1 | 0.4 | 0.4 | 0.6 | correctly classified inside margin |
| 3 | -1 | -0.2 | 0.2 | 0.8 | correctly classified inside margin |
| 4 | -1 | 0.5 | -0.5 | 1.5 | misclassified |

Slack variables satisfy \( \xi_i = \max(0, 1 - y_i f(x_i)) \), so they are the same quantity as per-sample hinge losses.

**Q3. Support vectors**
- Support vectors are the training samples closest to the decision boundary.
- They determine the margin and hyperplane.
- Non-support-vector samples usually do not change the model if removed.
- In bioinformatics, support vectors may correspond to ambiguous, noisy, heterogeneous, or borderline samples.

### Part II — Linear SVM on gene expression data

**Data preprocessing**
- Split: 70% train, 30% test.
- Scale features with `StandardScaler`.
- Scaling is necessary because SVM depends on distances and dot products.

**Linear SVM**
- Train `SVC(kernel='linear', C=1)`.
- Report training accuracy, test accuracy, and number of support vectors.
- Large support-vector fraction implies overlap, noise, or a difficult boundary.

**Feature importance**
- For linear SVM, `coef_` gives feature weights.
- Rank genes by absolute coefficient.
- Genes with large absolute coefficients are candidate biomarkers.

### Part III — Tuning C
- Use `C = {0.01, 0.1, 1, 10, 100}` with stratified 5-fold CV.
- Small C → stronger regularization, wider margin, more underfitting risk.
- Large C → less regularization, narrower margin, more overfitting risk.
- Choose the C with the highest validation accuracy.

### Part IV — RBF SVM
- Tune over `C = {0.1, 1, 10}` and `gamma = {0.001, 0.01, 0.1}`.
- Compare test accuracy to linear SVM.
- RBF can perform better if the true boundary is nonlinear.

### Part V — Support vector analysis
- Compute support-vector fraction.
- Plot decision scores.
- Identify samples with \(y f(x) < 1\) as on/inside the margin.
- These may represent biologically ambiguous or noisy cases.

### Part VI — Critical thinking
- SVM works well for high-dimensional omics because margin control regularizes the classifier.
- Kernel SVM can become expensive on very large datasets.
- Logistic regression may be preferable for calibrated probabilities, simpler interpretation, and very large sample sizes.

## Files included
- `svm_homework_solution_notebook.ipynb`: complete worked notebook
- `gene_expression_binary.csv`: synthetic dataset used by the notebook
