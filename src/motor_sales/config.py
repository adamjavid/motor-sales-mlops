"""Central configuration: schema, feature lists and model settings."""

from dataclasses import dataclass, field

TARGET = "Sale"
DATE_COL = "Date"

NUMERIC_FEATURES = [
    "Driver_Age",
    "Vehicle_Value",
    "Tax",
    "Price",
    "Vehicle_Mileage",
    "Credit_Score",
    "Licence_Length_Years",
]
CATEGORICAL_FEATURES = ["Marital_Status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Expected columns and their allowed ranges, used by data validation.
SCHEMA = {
    "Driver_Age": (17, 100),
    "Vehicle_Value": (0, 100_000),
    "Tax": (0, 500),
    "Price": (0, 5_000),
    "Vehicle_Mileage": (0, 100_000),
    "Credit_Score": (0, 9_999),
    "Licence_Length_Years": (0, 80),
}
MARITAL_STATUS_VALUES = {"M", "S", "D"}

# Sentinel value used in the source data for an unknown credit score.
CREDIT_SCORE_SENTINEL = 9_999

# Quotes in this predicted-probability band are flagged for follow-up.
FOLLOW_UP_BAND = (0.40, 0.60)


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    # Minimum test AUC a model must reach before it is saved.
    min_auc: float = 0.70
    models: list[str] = field(
        default_factory=lambda: ["logistic_regression", "random_forest"]
    )
