"""Wrapper serializável que fecha o contrato de inferência do modelo.

O wrapper guarda, junto do estimador, o schema de features (ordem, tipos e
categorias válidas). Assim, qualquer consumidor — simulação, API FastAPI,
testes — envia um DataFrame "cru" e o wrapper garante a coerção determinística
de dtypes antes do predict, eliminando skew treino/serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ChurnModel:
    estimator: Any
    schema: dict
    version: str = "0"
    trained_at: str = ""
    metrics: dict = field(default_factory=dict)
    threshold: float = 0.5

    @property
    def feature_names(self) -> list[str]:
        return self.schema["numeric_features"] + self.schema["categorical_features"]

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coage o DataFrame de entrada ao contrato de features do treino."""
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"features ausentes na entrada: {missing}")
        out = df[self.feature_names].copy()
        for col in self.schema["numeric_features"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in self.schema["categorical_features"]:
            cats = self.schema["categories"][col]
            out[col] = pd.Categorical(out[col].astype(str), categories=cats)
        return out

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.prepare(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def save(self, model_dir: str | Path) -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "model.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> ChurnModel:
        return joblib.load(Path(model_dir) / "model.joblib")

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")
