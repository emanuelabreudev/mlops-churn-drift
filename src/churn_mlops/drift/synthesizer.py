"""Sintetizadores de dados tabulares para simular o fluxo de produção.

Dois engines intercambiáveis (config: simulation.synthesizer):

- ``internal`` (default): copula gaussiana própria — marginais empíricas
  (numéricas) e frequências (categóricas) acopladas por uma matriz de
  correlação no espaço normal. É o mesmo princípio do
  ``GaussianCopulaSynthesizer`` do SDV, sem arrastar a dependência de
  PyTorch (~2 GB) para dentro do pipeline e da CI. Ver ADR-0003.
- ``sdv``: usa o SDV da especificação original (instale com
  ``uv sync --extra sdv``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def enforce_structural_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Impõe as restrições estruturais do domínio que a copula não captura.

    A copula modela correlações, não regras rígidas — pode amostrar, por
    exemplo, fibra óptica com add-on "No internet service". Regras:
    - InternetService == "No"  <=> add-ons == "No internet service";
    - PhoneService == "No"     <=> MultipleLines == "No phone service".
    """
    out = df.copy()
    no_internet = out["InternetService"] == "No"
    for col in INTERNET_ADDON_COLS:
        if col in out.columns:
            out.loc[no_internet, col] = "No internet service"
            fix = ~no_internet & (out[col] == "No internet service")
            out.loc[fix, col] = "No"
    if "PhoneService" in out.columns and "MultipleLines" in out.columns:
        no_phone = out["PhoneService"] == "No"
        out.loc[no_phone, "MultipleLines"] = "No phone service"
        fix = ~no_phone & (out["MultipleLines"] == "No phone service")
        out.loc[fix, "MultipleLines"] = "No"
    return out


class GaussianCopulaSynthesizer:
    """Copula gaussiana para dados mistos (numéricos + categóricos)."""

    def __init__(self, schema: dict, seed: int = 42):
        self.schema = schema
        self.seed = seed
        self._numeric: dict[str, np.ndarray] = {}
        self._categories: dict[str, tuple[list[str], np.ndarray]] = {}
        self._columns: list[str] = []
        self._corr: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> GaussianCopulaSynthesizer:
        rng = np.random.default_rng(self.seed)
        self._columns = self.schema["numeric_features"] + self.schema["categorical_features"]
        z_cols = []

        for col in self.schema["numeric_features"]:
            vals = np.sort(df[col].to_numpy(dtype=float))
            self._numeric[col] = vals
            ranks = stats.rankdata(df[col], method="average") / (len(df) + 1)
            z_cols.append(stats.norm.ppf(ranks))

        for col in self.schema["categorical_features"]:
            freq = df[col].astype(str).value_counts(normalize=True)
            cats, probs = list(freq.index), freq.to_numpy()
            self._categories[col] = (cats, probs)
            # posição uniforme dentro do intervalo acumulado da categoria
            cum = np.concatenate([[0.0], np.cumsum(probs)])
            idx = pd.Categorical(df[col].astype(str), categories=cats).codes
            u = cum[idx] + rng.uniform(0, 1, len(df)) * probs[idx]
            z_cols.append(stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9)))

        z = np.column_stack(z_cols)
        corr = np.corrcoef(z, rowvar=False)
        # regularização leve garante matriz positiva-definida para a amostragem
        self._corr = 0.995 * corr + 0.005 * np.eye(corr.shape[0])
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if self._corr is None:
            raise RuntimeError("synthesizer não ajustado; chame fit() antes")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        z = rng.multivariate_normal(np.zeros(len(self._columns)), self._corr, size=n, method="cholesky")
        u = stats.norm.cdf(z)

        data = {}
        for j, col in enumerate(self._columns):
            if col in self._numeric:
                data[col] = np.quantile(self._numeric[col], u[:, j])
            else:
                cats, probs = self._categories[col]
                cum = np.cumsum(probs)
                idx = np.clip(np.searchsorted(cum, u[:, j], side="right"), 0, len(cats) - 1)
                data[col] = np.array(cats, dtype=object)[idx]
        df = pd.DataFrame(data)[self._columns]
        df["tenure"] = df["tenure"].round().clip(lower=0)
        return enforce_structural_consistency(df)


class SDVSynthesizer:
    """Adaptador fino sobre o sdv.single_table.GaussianCopulaSynthesizer."""

    def __init__(self, schema: dict, seed: int = 42):
        self.schema = schema
        self.seed = seed
        self._synth = None

    def fit(self, df: pd.DataFrame) -> SDVSynthesizer:
        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import GaussianCopulaSynthesizer as SDVGaussianCopula
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SDV não instalado. Rode `uv sync --extra sdv` ou use synthesizer: internal no config."
            ) from exc
        cols = self.schema["numeric_features"] + self.schema["categorical_features"]
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df[cols])
        for col in self.schema["categorical_features"]:
            metadata.update_column(col, sdtype="categorical")
        self._synth = SDVGaussianCopula(metadata)
        self._synth.fit(df[cols])
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:  # noqa: ARG002 - sdv controla seed global
        return self._synth.sample(num_rows=n)


def make_synthesizer(engine: str, schema: dict, seed: int):
    if engine == "internal":
        return GaussianCopulaSynthesizer(schema, seed)
    if engine == "sdv":
        return SDVSynthesizer(schema, seed)
    raise ValueError(f"engine desconhecido: {engine!r} (use 'internal' ou 'sdv')")
