import numpy as np
import pytest

from motor_sales.data import clean, split_xy
from motor_sales.pipeline import build_model


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_model_handles_missing_values(quotes, name):
    X, y = split_xy(clean(quotes))
    assert X.isna().any().any(), "fixture should contain missing values"
    proba = build_model(name).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_unseen_category_does_not_break_scoring(quotes):
    X, y = split_xy(clean(quotes))
    model = build_model("logistic_regression").fit(X, y)
    new = X.head(5).copy()
    new["Marital_Status"] = "W"
    assert len(model.predict_proba(new)) == 5


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("xgboost")


def test_training_is_reproducible(quotes):
    X, y = split_xy(clean(quotes))
    a = build_model("random_forest", random_state=1).fit(X, y).predict_proba(X)
    b = build_model("random_forest", random_state=1).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)
