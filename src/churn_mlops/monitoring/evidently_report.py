"""Relatórios visuais de drift via Evidently AI (HTML).

O Evidently gera os relatórios interativos por lote; a *decisão* de retreinar
usa churn_mlops.monitoring.metrics (ver ADR-0004). O import é adiado e a
falha é não-fatal: se a API do Evidently mudar, o pipeline continua operando
e apenas perde o artefato visual.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, path: str | Path
) -> Path | None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current, reference)
        snapshot.save_html(str(path))
        return path
    except Exception as exc:  # pragma: no cover - degradação graciosa
        print(f"[evidently] relatório HTML indisponível ({type(exc).__name__}: {exc})")
        return None
