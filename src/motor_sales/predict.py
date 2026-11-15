"""Batch scoring and follow-up prioritisation."""

from pathlib import Path

import joblib
import pandas as pd

from motor_sales.config import FEATURES, FOLLOW_UP_BAND
from motor_sales.data import clean, validate


def load_model(path: str | Path):
    return joblib.load(path)


def score(model, df: pd.DataFrame) -> pd.DataFrame:
    """Add purchase probability and a follow-up flag to each quote."""
    validate(df, require_target=False)
    out = df.copy()
    out["purchase_probability"] = model.predict_proba(clean(df)[FEATURES])[:, 1].round(4)
    low, high = FOLLOW_UP_BAND
    out["follow_up"] = out["purchase_probability"].between(low, high)
    return out
