"""Loading, validation and light cleaning of quote data."""

from pathlib import Path

import numpy as np
import pandas as pd

from motor_sales.config import (
    CREDIT_SCORE_SENTINEL,
    DATE_COL,
    FEATURES,
    MARITAL_STATUS_VALUES,
    SCHEMA,
    TARGET,
)


class DataValidationError(ValueError):
    """Raised when input data does not match the expected schema."""


def load_quotes(path: str | Path) -> pd.DataFrame:
    """Load quote data from CSV or Excel."""
    path = Path(path)
    if path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=[DATE_COL])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Check columns, types and value ranges. Raises DataValidationError."""
    required = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if df.empty:
        raise DataValidationError("Dataset is empty")

    for col, (low, high) in SCHEMA.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"{col} must be numeric")
        values = df[col].dropna()
        out_of_range = values[(values < low) | (values > high)]
        if not out_of_range.empty:
            raise DataValidationError(
                f"{col} has {len(out_of_range)} values outside [{low}, {high}]"
            )

    unknown = set(df["Marital_Status"].dropna()) - MARITAL_STATUS_VALUES
    if unknown:
        raise DataValidationError(f"Unexpected Marital_Status values: {unknown}")

    if require_target and not set(df[TARGET].unique()) <= {0, 1}:
        raise DataValidationError(f"{TARGET} must be binary 0/1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy. Imputation happens inside the model pipeline
    (fitted on training data only) to avoid data leakage."""
    out = df.copy()
    # The source data uses 9999 to mean "unknown" credit score.
    out.loc[out["Credit_Score"] == CREDIT_SCORE_SENTINEL, "Credit_Score"] = np.nan
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
