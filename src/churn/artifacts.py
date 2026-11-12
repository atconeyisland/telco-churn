from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn

from churn import config


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def save_model(
    pipeline,
    *,
    model_name: str,
    strategy: str,
    threshold: float,
    metrics: dict,
    params: dict,
) -> None:
    """Persist the fitted pipeline plus a model card (threshold, metrics, versions)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH, compress=3)
    card = {
        "model": model_name,
        "imbalance_strategy": strategy,
        "threshold": float(threshold),
        "test_metrics": metrics,
        "best_params": params,
        "feature_columns": config.FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.MODEL_CARD_PATH.write_text(
        json.dumps(card, indent=2, default=_json_default), encoding="utf-8"
    )


def load_model():
    """Return (pipeline, model_card dict)."""
    if not config.MODEL_PATH.exists() or not config.MODEL_CARD_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run the notebook to train and save the model."
        )
    pipeline = joblib.load(config.MODEL_PATH)
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    return pipeline, card
