from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from churn import config

STRATEGIES = ("none", "weights", "smote")
BASELINE_NAMES = ("Dummy", "Logistic Regression", "Decision Tree", "KNN")
TUNABLE_NAMES = ("Random Forest", "Gradient Boosting", "XGBoost")

# Estimators that natively support class weighting
WEIGHT_CAPABLE = frozenset({"Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"})

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__max_features": ["sqrt"],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
        "clf__subsample": [0.8, 1.0],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [3, 5],
        "clf__colsample_bytree": [0.7, 1.0],
    },
}


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot categoricals. Fitted only inside pipelines."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC), ("cat", categorical, config.CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def is_supported(name: str, strategy: str) -> bool:
    return strategy != "weights" or name in WEIGHT_CAPABLE


def make_estimator(name: str, strategy: str = "none", scale_pos_weight: float = 1.0):
    rs = config.RANDOM_STATE
    class_weight = "balanced" if strategy == "weights" else None

    if name == "Dummy":
        return DummyClassifier(strategy="prior")
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=rs)
    if name == "Decision Tree":
        return DecisionTreeClassifier(class_weight=class_weight, random_state=rs)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight=class_weight, n_jobs=1, random_state=rs
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=rs)
    if name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=rs,
            verbosity=0,
            scale_pos_weight=scale_pos_weight if strategy == "weights" else 1.0,
        )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(
    name: str, strategy: str = "none", scale_pos_weight: float = 1.0
) -> ImbPipeline:
    """preprocess -> [SMOTE] -> classifier. SMOTE runs only on training folds."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if not is_supported(name, strategy):
        raise ValueError(f"{name} does not support class weights; use 'none' or 'smote'.")

    steps = [("pre", build_preprocessor())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("clf", make_estimator(name, strategy, scale_pos_weight)))
    return ImbPipeline(steps)


def baseline_models() -> dict[str, ImbPipeline]:
    return {name: build_pipeline(name) for name in BASELINE_NAMES}
