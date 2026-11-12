from __future__ import annotations

import time

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from churn import config, models

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

GRID_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": "recall",
}


def _cv() -> StratifiedKFold:
    return StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def cv_summary(estimators: dict, X, y) -> pd.DataFrame:
    """Stratified k-fold CV table formatted as 'mean ± std' per metric."""
    rows = {}
    for label, estimator in estimators.items():
        scores = cross_validate(clone(estimator), X, y, cv=_cv(), scoring=CV_SCORING, n_jobs=1)
        rows[label] = {
            metric: f"{scores[f'test_{metric}'].mean():.4f} ± {scores[f'test_{metric}'].std():.4f}"
            for metric in CV_SCORING
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")


def tune(name: str, strategy: str, X, y, n_jobs: int = 1) -> GridSearchCV:
    """GridSearchCV over the model's grid; refits on ROC-AUC."""
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    pipeline = models.build_pipeline(name, strategy, scale_pos_weight=n_neg / n_pos)
    search = GridSearchCV(
        pipeline,
        models.PARAM_GRIDS[name],
        scoring=GRID_SCORING,
        refit="roc_auc",
        cv=_cv(),
        n_jobs=n_jobs,
        error_score="raise",
    )
    return search.fit(X, y)


def tune_all(X, y, names=models.TUNABLE_NAMES, strategies=models.STRATEGIES) -> dict:
    """Run every supported (model, imbalance strategy) combination."""
    searches = {}
    for name in names:
        for strategy in strategies:
            if not models.is_supported(name, strategy):
                continue
            start = time.perf_counter()
            search = tune(name, strategy, X, y)
            searches[(name, strategy)] = search
            print(
                f"{name:<18} {strategy:<8} cv_roc_auc={search.best_score_:.4f} "
                f"({time.perf_counter() - start:.0f}s)"
            )
    return searches


def summarize(searches: dict) -> pd.DataFrame:
    """Best CV scores per (model, strategy), sorted by CV ROC-AUC."""
    rows = []
    for (name, strategy), search in searches.items():
        i = search.best_index_
        res = search.cv_results_
        rows.append(
            {
                "model": name,
                "strategy": strategy,
                "cv_roc_auc": res["mean_test_roc_auc"][i],
                "cv_pr_auc": res["mean_test_pr_auc"][i],
                "cv_f1": res["mean_test_f1"][i],
                "cv_recall": res["mean_test_recall"][i],
                "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def best_per_model(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values("cv_roc_auc", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def collect_best(searches: dict, summary: pd.DataFrame) -> dict:
    """{'Random Forest (smote)': fitted pipeline, ...} - best strategy per model."""
    return {
        f"{row.model} ({row.strategy})": searches[(row.model, row.strategy)].best_estimator_
        for row in best_per_model(summary).itertuples()
    }


def select_final(summary: pd.DataFrame) -> tuple[str, str]:
    """Final model = highest CV ROC-AUC. Never selected on the test set."""
    top = summary.sort_values("cv_roc_auc", ascending=False).iloc[0]
    return top["model"], top["strategy"]
