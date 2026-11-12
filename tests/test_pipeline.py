from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config, data, models, thresholds


def make_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame that mimics the raw CSV (TotalCharges as text with a blank for tenure 0)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 73, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["No phone service", "No", "Yes"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            **{c: rng.choice(["Yes", "No", "No internet service"], n) for c in config.ADDON_COLS},
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(18, 120, n).round(2),
        }
    )
    df.loc[0, "tenure"] = 0
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["Churn"] = np.where(np.arange(n) % 4 == 0, "Yes", "No")
    return df


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_raw(tmp_path / "nope.csv")


def test_clean_data_fixes_total_charges():
    clean = data.clean_data(make_frame())
    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    assert clean.loc[0, "TotalCharges"] == 0.0
    assert not clean["TotalCharges"].isna().any()


def test_clean_data_maps_senior_citizen():
    clean = data.clean_data(make_frame())
    assert set(clean["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_add_features():
    df = data.clean_data(make_frame(5))
    df.loc[:, config.ADDON_COLS] = "Yes"
    df["PaymentMethod"] = [
        "Bank transfer (automatic)",
        "Mailed check",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    out = data.add_features(df)
    assert (out["n_addons"] == len(config.ADDON_COLS)).all()
    assert out["is_autopay"].tolist() == [1, 0, 1, 0, 0]


def test_prepare_features_schema():
    features = data.prepare_features(make_frame())
    assert list(features.columns) == config.FEATURE_COLUMNS
    assert not features.isna().any().any()


def test_encode_target_rejects_unknown_values():
    with pytest.raises(ValueError):
        data.encode_target(pd.Series(["Yes", "Maybe"]))


def test_split_is_stratified_70_30():
    df = data.add_features(data.clean_data(make_frame()))
    X_train, X_test, y_train, y_test = data.make_splits(df)
    assert len(X_test) == 60 and len(X_train) == 140
    assert abs(y_train.mean() - y_test.mean()) < 0.02


@pytest.mark.parametrize(
    "name,strategy",
    [
        ("Logistic Regression", "none"),
        ("Logistic Regression", "weights"),
        ("Random Forest", "smote"),
        ("Gradient Boosting", "none"),
        ("XGBoost", "weights"),
        ("KNN", "smote"),
    ],
)
def test_pipelines_fit_and_predict(name, strategy):
    df = data.add_features(data.clean_data(make_frame()))
    X_train, X_test, y_train, _ = data.make_splits(df)
    pipe = models.build_pipeline(name, strategy, scale_pos_weight=3.0).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_unsupported_weight_combo_raises():
    with pytest.raises(ValueError):
        models.build_pipeline("KNN", "weights")


def test_best_threshold_on_separable_data():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t, score = thresholds.best_threshold(y, p)
    assert 0.2 < t <= 0.8
    assert score == pytest.approx(1.0)
