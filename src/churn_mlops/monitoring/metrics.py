"""Métricas de data drift e motor de decisão do gatilho de retreinamento.

As decisões (retreinar ou não) usam esta implementação própria de PSI/KS/χ²
— determinística, com limiares explícitos e testável unitariamente. O
Evidently AI é usado em paralelo para os relatórios visuais HTML (ver
evidently_report.py e ADR-0004): separar "decisão" de "visualização" evita
acoplar o gatilho de produção a mudanças de API de uma biblioteca externa.

Convenção de limiares de PSI (Siddiqi, 2017 — credit scoring):
  PSI < 0.10          distribuição estável
  0.10 <= PSI < 0.25  drift moderado (alerta)
  PSI >= 0.25         drift significativo (ação)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI para variável numérica, com bins de quantis da referência."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variável quase constante na referência
        edges = np.array([-np.inf, np.median(reference), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
    cur_frac = np.histogram(current, bins=edges)[0] / max(len(current), 1)
    ref_frac = np.clip(ref_frac, EPS, None)
    cur_frac = np.clip(cur_frac, EPS, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.unique()) | set(current.unique()))
    ref_frac = reference.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    cur_frac = current.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    ref_frac = np.clip(ref_frac, EPS, None)
    cur_frac = np.clip(cur_frac, EPS, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    return float(stats.ks_2samp(reference, current).pvalue)


def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    cur_counts = current.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    expected = ref_counts / ref_counts.sum() * cur_counts.sum()
    mask = expected > 0
    if mask.sum() < 2:
        return 1.0
    return float(stats.chisquare(cur_counts[mask], expected[mask]).pvalue)


@dataclass
class FeatureDrift:
    feature: str
    kind: str  # numeric | categorical
    psi: float
    p_value: float
    status: str  # ok | moderate | critical


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    share_drifted: float
    max_psi: float
    mean_psi: float
    dataset_drift: bool
    severity: str  # OK | WARNING | CRITICAL
    thresholds: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "share_drifted": self.share_drifted,
            "max_psi": self.max_psi,
            "mean_psi": self.mean_psi,
            "dataset_drift": self.dataset_drift,
            "severity": self.severity,
        }


def evaluate_drift(
    reference: pd.DataFrame, current: pd.DataFrame, schema: dict, drift_cfg: dict
) -> DriftReport:
    """Compara um lote com a referência de treino, feature a feature."""
    moderate, critical = drift_cfg["psi_moderate"], drift_cfg["psi_critical"]
    bins = drift_cfg.get("psi_bins", 10)
    feats: list[FeatureDrift] = []

    for col in schema["numeric_features"]:
        ref = reference[col].to_numpy(dtype=float)
        cur = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy(dtype=float)
        psi = psi_numeric(ref, cur, bins)
        p = ks_pvalue(ref, cur)
        feats.append(FeatureDrift(col, "numeric", psi, p, _status(psi, moderate, critical)))

    for col in schema["categorical_features"]:
        ref, cur = reference[col].astype(str), current[col].astype(str)
        psi = psi_categorical(ref, cur)
        p = chi2_pvalue(ref, cur)
        feats.append(FeatureDrift(col, "categorical", psi, p, _status(psi, moderate, critical)))

    psis = np.array([f.psi for f in feats])
    share = float(np.mean([f.status != "ok" for f in feats]))
    any_critical = any(f.status == "critical" for f in feats)
    dataset_drift = any_critical or share >= drift_cfg["drifted_share_threshold"]
    severity = "CRITICAL" if dataset_drift else ("WARNING" if share > 0 else "OK")

    return DriftReport(
        features=feats,
        share_drifted=share,
        max_psi=float(psis.max()),
        mean_psi=float(psis.mean()),
        dataset_drift=dataset_drift,
        severity=severity,
        thresholds={"psi_moderate": moderate, "psi_critical": critical},
    )


def _status(psi: float, moderate: float, critical: float) -> str:
    if psi >= critical:
        return "critical"
    if psi >= moderate:
        return "moderate"
    return "ok"
