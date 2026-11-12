from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from churn import config
from churn.thresholds import metrics_by_threshold

PALETTE = {"No": "#4C72B0", "Yes": "#DD5555"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold"})


def save_fig(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")


def _finish(fig, name: str | None) -> None:
    fig.tight_layout()
    if name:
        save_fig(fig, name)
    plt.show()


def _grid(n: int, ncols: int, cell_w: float, cell_h: float):
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


# ------------------------------------------------------------------ EDA
def plot_target_distribution(target: pd.Series, name: str | None = None) -> None:
    counts = target.value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set(title="Target distribution", xlabel="Churn", ylabel="Customers")
    _finish(fig, name)


def plot_numeric_distributions(df, cols, hue: str = "Churn", name: str | None = None) -> None:
    fig, axes = _grid(len(cols), len(cols), 5, 3.8)
    for ax, col in zip(axes, cols):
        sns.histplot(
            data=df, x=col, hue=hue, kde=True, stat="density", common_norm=False,
            palette=PALETTE, alpha=0.35, ax=ax,
        )
        ax.set_title(col)
    _finish(fig, name)


def plot_boxplots(df, cols, hue: str = "Churn", name: str | None = None) -> None:
    fig, axes = _grid(len(cols), len(cols), 4.5, 4)
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=hue, y=col, hue=hue, palette=PALETTE, legend=False, ax=ax)
        ax.set_title(col)
    _finish(fig, name)


def plot_churn_rate_grid(
    df, cols, target: str = "Churn", ncols: int = 3, name: str | None = None
) -> None:
    base = (df[target] == "Yes").mean()
    fig, axes = _grid(len(cols), ncols, 5, 3.2)
    for ax, col in zip(axes, cols):
        rate = df.groupby(col, observed=True)[target].apply(lambda s: (s == "Yes").mean())
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            rate = rate.sort_values(ascending=False)
        sns.barplot(x=rate.values, y=rate.index.astype(str), orient="h", color="#DD5555", ax=ax)
        ax.axvline(base, color="black", ls="--", lw=1)
        ax.set(title=col, xlabel="Churn rate", ylabel="")
        ax.set_xlim(0, max(0.6, rate.max() * 1.1))
    fig.suptitle(f"Churn rate by category (dashed = overall {base:.1%})", y=1.0, fontweight="bold")
    _finish(fig, name)


def plot_correlation_heatmap(corr: pd.DataFrame, name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax
    )
    ax.set_title("Correlation matrix")
    _finish(fig, name)


def plot_ranked_bar(
    series: pd.Series, title: str, xlabel: str, name: str | None = None, top: int = 15
) -> None:
    ranked = series.reindex(series.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    colors = np.where(ranked.values >= 0, "#DD5555", "#4C72B0")
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(ranked) + 1.5))
    ax.barh(ranked.index.astype(str), ranked.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title=title, xlabel=xlabel)
    _finish(fig, name)


def plot_heatmap(pivot: pd.DataFrame, title: str, name: str | None = None, fmt: str = ".3f") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 0.9 * len(pivot) + 1.5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="YlGnBu", ax=ax)
    ax.set(title=title, xlabel="", ylabel="")
    _finish(fig, name)


# ------------------------------------------------------------------ evaluation
def plot_confusion_matrices(
    models: dict, X, y, thresholds: dict | None = None, name: str | None = None
) -> None:
    thresholds = thresholds or {}
    fig, axes = _grid(len(models), min(len(models), 4), 4, 3.8)
    for ax, (label, model) in zip(axes, models.items()):
        t = thresholds.get(label, 0.5)
        pred = (model.predict_proba(X)[:, 1] >= t).astype(int)
        sns.heatmap(
            confusion_matrix(y, pred, labels=[0, 1]), annot=True, fmt=",d", cmap="Blues",
            cbar=False, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax,
        )
        ax.set(title=f"{label}\n(threshold {t:.2f})", xlabel="Predicted", ylabel="Actual")
    _finish(fig, name)


def plot_roc_curves(models: dict, X, y, name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, model in models.items():
        prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc_score(y, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    _finish(fig, name)


def plot_pr_curves(models: dict, X, y, name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, model in models.items():
        prob = model.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, prob)
        ax.plot(recall, precision, lw=2, label=f"{label} (AP={average_precision_score(y, prob):.3f})")
    ax.axhline(float(np.mean(y)), color="k", ls="--", lw=1, label="Prevalence")
    ax.set(title="Precision-recall curves (test set)", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="upper right", fontsize=8)
    _finish(fig, name)


def plot_calibration(models: dict, X, y, n_bins: int = 10, name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, model in models.items():
        frac_pos, mean_pred = calibration_curve(
            y, model.predict_proba(X)[:, 1], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.set(
        title="Calibration (test set)",
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
    )
    ax.legend(fontsize=8)
    _finish(fig, name)


def plot_threshold_tradeoff(
    y_true, y_prob, marks: dict[str, float], name: str | None = None
) -> None:
    table = metrics_by_threshold(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in [("precision", "Precision"), ("recall", "Recall"), ("fbeta", "F1")]:
        ax.plot(table["threshold"], table[col], lw=2, label=label)
    for i, (label, t) in enumerate(marks.items()):
        ax.axvline(t, ls="--", lw=1, color=f"C{3 + i}", label=f"{label} ({t:.2f})")
    ax.set(
        title="Threshold trade-off (out-of-fold, train)",
        xlabel="Decision threshold",
        ylabel="Score",
    )
    ax.legend(fontsize=8)
    _finish(fig, name)


def plot_cost_curve(curve: pd.DataFrame, marks: dict[str, float], name: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["threshold"], curve["cost_per_customer"], lw=2)
    for i, (label, t) in enumerate(marks.items()):
        ax.axvline(t, ls="--", lw=1, color=f"C{1 + i}", label=f"{label} ({t:.2f})")
    ax.set(
        title=(
            f"Expected cost per customer "
            f"(FN=${config.COST_FALSE_NEGATIVE:,.0f}, FP=${config.COST_FALSE_POSITIVE:,.0f})"
        ),
        xlabel="Decision threshold",
        ylabel="Cost per customer ($)",
    )
    ax.legend(fontsize=8)
    _finish(fig, name)


def plot_shap_summary(values, X: pd.DataFrame, name: str | None = None) -> None:
    import shap

    shap.summary_plot(values, X, show=False, max_display=15)
    _finish(plt.gcf(), name)
