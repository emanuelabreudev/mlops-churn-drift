"""API de serving do modelo de churn (FastAPI).

O modelo é carregado do diretório de artefatos (env ``MODEL_DIR``, default
``artifacts/model``) — o mesmo formato exportado pelo treino e pelos
retreinos. ``POST /reload`` recarrega o artefato sem derrubar o processo,
permitindo promover uma nova versão com zero downtime de deploy.

``GET /metrics`` expõe contadores no formato de texto do Prometheus, de modo
que um Grafana possa ser plugado sem dependências extras (ver ADR-0005).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from churn_mlops import __version__
from churn_mlops.config import resolve
from churn_mlops.models.wrapper import ChurnModel
from churn_mlops.serving.schemas import ModelInfo, PredictRequest, PredictResponse

state: dict = {"model": None, "requests": 0, "predictions": 0, "churn_flagged": 0, "latency_sum": 0.0}


def model_dir() -> Path:
    return resolve(os.environ.get("MODEL_DIR", "artifacts/model"))


def load_model() -> None:
    state["model"] = ChurnModel.load(model_dir())


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Churn Prediction API",
    version=__version__,
    description="Serving do pipeline de MLOps com monitoramento de data drift",
    lifespan=lifespan,
)


def get_model() -> ChurnModel:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")
    return state["model"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    m = get_model()
    return ModelInfo(
        model_version=m.version,
        trained_at=m.trained_at,
        threshold=m.threshold,
        features=m.feature_names,
        metrics=m.metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    m = get_model()
    start = time.perf_counter()
    df = pd.DataFrame([c.model_dump() for c in req.customers])
    proba = m.predict_proba(df)
    preds = (proba >= m.threshold).astype(bool)

    state["requests"] += 1
    state["predictions"] += len(df)
    state["churn_flagged"] += int(preds.sum())
    state["latency_sum"] += time.perf_counter() - start
    return PredictResponse(
        predictions=[
            {"churn_probability": round(float(p), 6), "churn_predicted": bool(y)}
            for p, y in zip(proba, preds, strict=True)
        ],
        model_version=m.version,
        threshold=m.threshold,
    )


@app.post("/reload")
def reload_model() -> dict:
    load_model()
    return {"status": "reloaded", "model_version": get_model().version}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    m = state["model"]
    lines = [
        "# HELP churn_api_requests_total Total de chamadas ao /predict",
        "# TYPE churn_api_requests_total counter",
        f"churn_api_requests_total {state['requests']}",
        "# HELP churn_api_predictions_total Total de clientes pontuados",
        "# TYPE churn_api_predictions_total counter",
        f"churn_api_predictions_total {state['predictions']}",
        "# HELP churn_api_churn_flagged_total Predições positivas de churn",
        "# TYPE churn_api_churn_flagged_total counter",
        f"churn_api_churn_flagged_total {state['churn_flagged']}",
        "# HELP churn_api_latency_seconds_sum Latência acumulada do /predict",
        "# TYPE churn_api_latency_seconds_sum counter",
        f"churn_api_latency_seconds_sum {state['latency_sum']:.6f}",
        "# HELP churn_model_info Versão do modelo em produção",
        "# TYPE churn_model_info gauge",
        f'churn_model_info{{version="{m.version if m else "none"}"}} 1',
    ]
    return "\n".join(lines) + "\n"
