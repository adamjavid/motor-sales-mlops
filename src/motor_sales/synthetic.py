"""Generate a synthetic quote dataset with the same schema as the original.

The original case-study data is not published. This generator produces
realistic-looking data so the pipeline, tests and CI can run end to end.
"""

import numpy as np
import pandas as pd

from motor_sales.config import FEATURES


def generate(n_rows: int = 10_000, seed: int = 42, missing_rate: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(30, 7.5, n_rows).round(), 18, 74)
    licence = np.clip((age - 17) * rng.uniform(0.2, 0.6, n_rows), 0, 16).round(2)
    vehicle_value = rng.choice(np.arange(2000, 11000, 1000).tolist() + [15000], n_rows)
    mileage = np.clip(rng.normal(7000, 1350, n_rows).round(-3), 1000, 13000)
    credit = np.clip(rng.normal(450, 120, n_rows), 50, 999).round(2)
    credit[rng.random(n_rows) < 0.01] = 9999  # sentinel for unknown
    tax = np.clip(rng.normal(48, 17, n_rows), 0, 100).round(2)
    price = np.clip(300 + tax * 5 + rng.normal(0, 60, n_rows), 100, 1000).round(2)
    marital = rng.choice(["M", "S", "D"], n_rows, p=[0.5, 0.4, 0.1])
    dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 730, n_rows), "D")

    # Purchase propensity with a known structure, so model quality can be tested.
    known_credit = np.where(credit == 9999, 450, credit)
    logit = (
        1.0
        + 0.9 * (tax - 48) / 17
        - 0.6 * (age - 30) / 7.5
        - 0.3 * (licence - 5) / 3
        + 0.2 * (known_credit - 450) / 120
        + 0.1 * (marital == "M")
    )
    sale = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "Driver_Age": age,
            "Vehicle_Value": vehicle_value.astype(float),
            "Tax": tax,
            "Price": price,
            "Vehicle_Mileage": mileage,
            "Credit_Score": credit,
            "Licence_Length_Years": licence,
            "Date": dates,
            "Marital_Status": marital,
            "Sale": sale,
        }
    )

    # Introduce missing values like the real data.
    for col in FEATURES:
        mask = rng.random(n_rows) < missing_rate
        df.loc[mask, col] = np.nan
    return df
