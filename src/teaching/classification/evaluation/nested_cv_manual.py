import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def evaluate_auc_cv(model, X, y, cv):
    """Return mean AUC across folds for given model and CV splitter."""
    aucs = []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, proba))
    return float(np.mean(aucs))


def nested_cv_manual(
    X,
    y,
    outer_splits=5,
    inner_splits=3,
    random_state=42
):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    # A small grid (keep modest since manual)
    grid = []
    for max_depth in [None, 3, 5, 8]:
        for min_samples_leaf in [1, 2, 5]:
            for ccp_alpha in [0.0, 0.001, 0.01]:
                grid.append((max_depth, min_samples_leaf, ccp_alpha))

    outer_fold_aucs = []
    chosen_params = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

        # Inner loop: choose best params on outer training data only
        best_auc = -np.inf
        best_params = None

        for (max_depth, min_samples_leaf, ccp_alpha) in grid:
            model = DecisionTreeClassifier(
                random_state=random_state,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha
            )
            mean_auc = evaluate_auc_cv(model, X_train, y_train, inner_cv)

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = (max_depth, min_samples_leaf, ccp_alpha)

        # Retrain with best params on full outer training set
        final_model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=best_params[0],
            min_samples_leaf=best_params[1],
            ccp_alpha=best_params[2]
        )
        final_model.fit(X_train, y_train)

        # Outer test evaluation
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

        outer_fold_aucs.append(test_auc)
        chosen_params.append((best_auc, best_params))

        print(f"\nOuter fold {fold}/{outer_splits}")
        print(f"  Best inner AUC: {best_auc:.3f}")
        print(f"  Chosen params:  max_depth={best_params[0]}, "
              f"min_samples_leaf={best_params[1]}, ccp_alpha={best_params[2]}")
        print(f"  Outer test AUC: {test_auc:.3f}")

    outer_fold_aucs = np.array(outer_fold_aucs, dtype=float)

    print("\n===== Manual Nested CV Summary =====")
    print(f"Outer AUC mean ± std: {outer_fold_aucs.mean():.3f} ± {outer_fold_aucs.std(ddof=1):.3f}")

    return {
        "outer_auc_scores": outer_fold_aucs.tolist(),
        "chosen_params_per_fold": chosen_params
    }


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        weights=[0.55, 0.45],
        class_sep=1.0,
        random_state=42
    )

    nested_cv_manual(X, y, outer_splits=5, inner_splits=3, random_state=42)
