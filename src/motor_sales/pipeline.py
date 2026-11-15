"""Model pipelines. All preprocessing lives inside the pipeline so it is
fitted on training data only and travels with the saved model."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from motor_sales.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_model(name: str, random_state: int = 42) -> Pipeline:
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", clf)])
