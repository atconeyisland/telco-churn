from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
FIGURES_DIR = ROOT / "reports" / "figures"
CACHE_DIR = ROOT / "artifacts"

# ---------------------------------------------------------------- experiment
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

# ---------------------------------------------------------------- schema
TARGET = "Churn"
ID_COL = "customerID"

RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDON_COLS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Numeric model inputs = raw numerics + engineered features (see data.add_features)
NUMERIC = [*RAW_NUMERIC, "n_addons", "is_autopay"]
FEATURE_COLUMNS = [*NUMERIC, *CATEGORICAL]

# ---------------------------------------------------------------- business costs
# Illustrative assumptions used ONLY for the cost-based threshold analysis.
# Replace with real figures if available.
COST_FALSE_NEGATIVE = 500.0  # churner we failed to flag (lost customer value)
COST_FALSE_POSITIVE = 50.0   # non-churner we flagged (wasted retention offer)
