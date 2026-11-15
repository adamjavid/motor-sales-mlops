import numpy as np
import pytest

from motor_sales.config import FEATURES, TARGET
from motor_sales.data import DataValidationError, clean, validate


def test_synthetic_data_passes_validation(quotes):
    validate(quotes)


def test_missing_column_fails(quotes):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(quotes.drop(columns=["Price"]))


def test_out_of_range_value_fails(quotes):
    bad = quotes.copy()
    bad.loc[0, "Driver_Age"] = 5
    with pytest.raises(DataValidationError, match="Driver_Age"):
        validate(bad)


def test_unknown_category_fails(quotes):
    bad = quotes.copy()
    bad.loc[0, "Marital_Status"] = "X"
    with pytest.raises(DataValidationError, match="Marital_Status"):
        validate(bad)


def test_non_binary_target_fails(quotes):
    bad = quotes.copy()
    bad.loc[0, TARGET] = 2
    with pytest.raises(DataValidationError, match="binary"):
        validate(bad)


def test_scoring_data_does_not_need_target(quotes):
    validate(quotes[FEATURES], require_target=False)


def test_clean_replaces_credit_score_sentinel(quotes):
    df = quotes.copy()
    df.loc[0, "Credit_Score"] = 9999
    cleaned = clean(df)
    assert np.isnan(cleaned.loc[0, "Credit_Score"])
    assert (cleaned["Credit_Score"] != 9999).all()


def test_clean_does_not_modify_input(quotes):
    before = quotes.copy()
    clean(quotes)
    assert quotes.equals(before)
