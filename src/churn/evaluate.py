from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn import config


def positive_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_models(models: dict, X, y, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """One row of metrics per fitted model."""
    thresholds = thresholds or {}
    rows = {
        label: compute_metrics(y, positive_proba(model, X), thresholds.get(label, 0.5))
        for label, model in models.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")


def threshold_comparison(model, X, y, thresholds: dict[str, float]) -> pd.DataFrame:
    """Metrics and illustrative business cost for one model at several operating points."""
    prob = positive_proba(model, X)
    rows = {}
    for label, t in thresholds.items():
        m = compute_metrics(y, prob, t)
        m["cost_per_customer"] = (
            m["fn"] * config.COST_FALSE_NEGATIVE + m["fp"] * config.COST_FALSE_POSITIVE
        ) / len(y)
        rows[label] = m
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("operating point")


def decile_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Gains/lift table: decile 1 = highest predicted churn risk."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["decile"] = pd.qcut(
        frame["p"].rank(method="first", ascending=False),
        n_bins,
        labels=range(1, n_bins + 1),
    )
    table = frame.groupby("decile", observed=True).agg(
        customers=("y", "size"), churners=("y", "sum"), avg_prob=("p", "mean")
    )
    table["churn_rate"] = table["churners"] / table["customers"]
    table["lift"] = table["churn_rate"] / frame["y"].mean()
    table["cum_capture"] = table["churners"].cumsum() / frame["y"].sum()
    return table
