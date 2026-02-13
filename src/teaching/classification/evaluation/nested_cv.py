import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def nested_cv_decision_tree(
    X,
    y,
    outer_splits=5,
    inner_splits=3,
    random_state=42,
    tune_metric="roc_auc",
):
    """
    Nested CV for binary classification with DecisionTreeClassifier.

    Outer CV: unbiased performance estimate
    Inner CV: hyperparameter tuning (GridSearchCV)

    tune_metric: metric used for selecting hyperparameters ("roc_auc", "accuracy", "f1", etc.)
    """

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    pipe = Pipeline([
        ("clf", DecisionTreeClassifier(random_state=random_state))
    ])

    param_grid = {
        "clf__max_depth": [None, 2, 3, 4, 5, 8, 12],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
        "clf__ccp_alpha": [0.0, 0.001, 0.01],
    }

    results = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=tune_metric,
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Outer test evaluation
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        fold_acc = accuracy_score(y_test, y_pred)
        fold_auc = roc_auc_score(y_test, y_proba)
        fold_f1  = f1_score(y_test, y_pred)

        results.append({
            "fold": fold,
            "best_params": search.best_params_,
            "inner_best_score": float(search.best_score_),
            "outer_acc": float(fold_acc),
            "outer_auc": float(fold_auc),
            "outer_f1": float(fold_f1),
        })

        print(f"\nOuter fold {fold}/{outer_splits}")
        print("  Best params:", search.best_params_)
        print(f"  Inner best ({tune_metric}): {search.best_score_:.3f}")
        print(f"  Outer ACC: {fold_acc:.3f} | AUC: {fold_auc:.3f} | F1: {fold_f1:.3f}")

    # Summaries
    accs = np.array([r["outer_acc"] for r in results])
    aucs = np.array([r["outer_auc"] for r in results])
    f1s  = np.array([r["outer_f1"] for r in results])

    summary = {
        "outer_acc_mean": float(accs.mean()),
        "outer_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "outer_auc_mean": float(aucs.mean()),
        "outer_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "outer_f1_mean": float(f1s.mean()),
        "outer_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "fold_results": results,
    }

    print("\n===== Nested CV Summary (outer test folds) =====")
    print(f"ACC: {summary['outer_acc_mean']:.3f} ± {summary['outer_acc_std']:.3f}")
    print(f"AUC: {summary['outer_auc_mean']:.3f} ± {summary['outer_auc_std']:.3f}")
    print(f" F1: {summary['outer_f1_mean']:.3f} ± {summary['outer_f1_std']:.3f}")

    return summary


if __name__ == "__main__":
    # Demo data
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        weights=[0.55, 0.45],
        class_sep=1.0,
        random_state=42
    )

    nested_cv_decision_tree(X, y, outer_splits=5, inner_splits=3, tune_metric="roc_auc")