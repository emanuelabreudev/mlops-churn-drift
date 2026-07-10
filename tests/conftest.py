"""Fixtures com dados sintéticos em miniatura (mesmo schema do Telco Churn).

Os testes não dependem do dataset real: rodam offline e em CI sem download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_mlops.data.preprocess import build_schema, clean

CATS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(vals, n) for col, vals in CATS.items()})
    df.insert(0, "customerID", [f"C{i:05d}" for i in range(n)])
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(18, 118, n).round(2)
    total = (df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.9, 1.1, n)).round(2)
    df["TotalCharges"] = total.astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # replica os blanks do dataset real
    # churn correlacionado com contrato mensal e tenure baixo, mais ruído
    logit = (
        -1.2
        + 1.4 * (df["Contract"] == "Month-to-month")
        - 0.03 * df["tenure"]
        + 0.8 * (df["InternetService"] == "Fiber optic")
    )
    p = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.uniform(0, 1, n) < p, "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture(scope="session")
def schema(clean_df) -> dict:
    return build_schema(clean_df, "Churn")
