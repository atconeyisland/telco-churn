from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn import config

DEFAULT_GRID = np.linspace(0.02, 0.98, 193)


def out_of_fold_proba(model, X, y) -> np.ndarray:
    """Out-of-fold P(churn) on the training set - lets us tune a threshold without touching test."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=1)
    return proba[:, 1]


def metrics_by_threshold(y_true, y_prob, beta: float = 1.0, grid=None) -> pd.DataFrame:
    """Precision, recall, F-beta and confusion counts across a grid of thresholds."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_prob, dtype=float)
    ths = DEFAULT_GRID if grid is None else np.asarray(grid, dtype=float)

    pred = p[None, :] >= ths[:, None]
    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = (~pred & y).sum(axis=1)
    tn = (~pred & ~y).sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros(len(ths)), where=(tp + fp) > 0)
    recall = tp / max(int(y.sum()), 1)
    b2 = beta**2
    denom = b2 * precision + recall
    fbeta = np.divide((1 + b2) * precision * recall, denom, out=np.zeros(len(ths)), where=denom > 0)

    return pd.DataFrame(
        {
            "threshold": ths,
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    )


def best_threshold(y_true, y_prob, beta: float = 1.0) -> tuple[float, float]:
    """Threshold maximising F-beta (beta=1 -> F1, beta=2 -> recall-weighted)."""
    table = metrics_by_threshold(y_true, y_prob, beta=beta)
    i = table["fbeta"].idxmax()
    return float(table.loc[i, "threshold"]), float(table.loc[i, "fbeta"])


def cost_curve(y_true, y_prob, cost_fn: float, cost_fp: float, grid=None) -> pd.DataFrame:
    table = metrics_by_threshold(y_true, y_prob, grid=grid)
    table["total_cost"] = table["fn"] * cost_fn + table["fp"] * cost_fp
    table["cost_per_customer"] = table["total_cost"] / len(y_true)
    return table


def cost_optimal_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["total_cost"].idxmin(), "threshold"])
