"""Registro de modelos retreinados no MLflow Model Registry.

Cada retreinamento acionado pelo gatilho de drift vira uma nova versão do
modelo registrado, com o alias de produção movido para ela — o histórico de
versões no registry conta a história dos retreinos. Falhas de tracking não
derrubam a simulação (o modelo novo continua valendo em memória): em
produção real, indisponibilidade do MLflow não pode parar o serving.
"""

from __future__ import annotations

import mlflow

from churn_mlops.config import load_config, resolve
from churn_mlops.models.wrapper import ChurnModel


def register_retrained(cfg: dict, model: ChurnModel, day: int, n_train: int) -> str:
    reg_name = cfg["mlflow"]["registered_model"]
    try:
        with mlflow.start_run(run_name=f"auto-retrain-day{day:02d}"):
            mlflow.log_params({"trigger": "data_drift", "sim_day": day, "n_train": n_train})
            mlflow.log_metric("threshold", model.threshold)
            info = mlflow.lightgbm.log_model(
                model.estimator, artifact_path="model", registered_model_name=reg_name
            )
        version = str(getattr(info, "registered_model_version", "") or "")
        if version:
            mlflow.MlflowClient().set_registered_model_alias(
                reg_name, cfg["mlflow"]["production_alias"], version
            )
            return version
    except Exception as exc:  # pragma: no cover - tracking é melhor-esforço
        print(f"[retrain] registro MLflow falhou ({type(exc).__name__}: {exc}); seguindo sem registrar")
    return f"retrain-day{day}"


def export_production(cfg: dict | None = None, model: ChurnModel | None = None) -> None:
    """Exporta o modelo corrente para artifacts/model (consumido pelo serving)."""
    cfg = cfg or load_config()
    if model is not None:
        model.save(resolve(cfg["artifacts"]["model_dir"]))
