from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from churn import config


def feature_names(pipeline) -> np.ndarray:
    return pipeline.named_steps["pre"].get_feature_names_out()


def builtin_importance(pipeline) -> pd.Series:
    """Impurity/gain importance of the final estimator, per encoded feature."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise TypeError(f"{type(clf).__name__} has no feature_importances_")
    return pd.Series(clf.feature_importances_, index=feature_names(pipeline)).sort_values(
        ascending=False
    )


def permutation_table(pipeline, X, y, scoring: str = "roc_auc", n_repeats: int = 20) -> pd.DataFrame:
    """Model-agnostic importance on ORIGINAL columns (one-hot levels stay grouped)."""
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    return pd.DataFrame(
        {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
        index=X.columns,
    ).sort_values("importance_mean", ascending=False)


def compute_shap(pipeline, X, max_rows: int = 1000):
    """SHAP values (log-odds, churn class) for a tree-based final estimator."""
    import shap

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    sample = X.sample(min(len(X), max_rows), random_state=config.RANDOM_STATE)
    transformed = pd.DataFrame(
        pre.transform(sample), columns=pre.get_feature_names_out(), index=sample.index
    )
    values = shap.TreeExplainer(clf).shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", 2) == 3:
        values = values[:, :, 1]
    return values, transformed
