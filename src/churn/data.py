from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

REQUIRED_COLUMNS = {
    config.ID_COL,
    config.TARGET,
    *config.RAW_NUMERIC,
    *config.CATEGORICAL,
}


def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and validate that the expected schema is present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and save it as data/raw/Telco-Customer-Churn.csv"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and obvious data-quality issues. Works on raw rows with or without a target.

    - strips whitespace from column names and string values
    - TotalCharges: blank strings -> NaN -> float; customers with tenure == 0 have not
      been billed yet, so their TotalCharges is 0.0 (a deterministic rule, not learned)
    - SeniorCitizen: 0/1 -> "No"/"Yes" so every binary feature is treated consistently
    """
    out = df.copy()
    out.columns = out.columns.str.strip()

    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].str.strip()

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if "tenure" in out.columns:
            not_billed = out["TotalCharges"].isna() & (out["tenure"] == 0)
            out.loc[not_billed, "TotalCharges"] = 0.0

    if "SeniorCitizen" in out.columns and pd.api.types.is_numeric_dtype(out["SeniorCitizen"]):
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering (safe to apply before the split)."""
    out = df.copy()
    out["n_addons"] = (out[config.ADDON_COLS] == "Yes").sum(axis=1)
    out["is_autopay"] = (
        out["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    )
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer rows -> model-ready feature frame. Used by training and the app."""
    return add_features(clean_data(df))[config.FEATURE_COLUMNS]


def encode_target(series: pd.Series) -> pd.Series:
    """'Yes'/'No' -> 1/0 with validation."""
    s = series.astype(str).str.strip().str.lower()
    unexpected = set(s.unique()) - {"yes", "no"}
    if unexpected:
        raise ValueError(f"Unexpected target values: {sorted(unexpected)}")
    return (s == "yes").astype(int).rename("churn")


def make_splits(df: pd.DataFrame):
    """70/30 stratified split on a frame produced by add_features(clean_data(raw))."""
    X = df[config.FEATURE_COLUMNS]
    y = encode_target(df[config.TARGET])
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
