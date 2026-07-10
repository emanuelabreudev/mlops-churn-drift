"""Treinamento com baselines, múltiplas seeds, tracking MLflow e registro do modelo.

Protocolo experimental (ver Seção de Metodologia do README):
- Split fixo 70/15/15 (feito no preprocess, seed do config) — o conjunto de
  teste nunca participa de fit ou de escolha de threshold.
- Para cada seed em ``n_seeds``: treina Regressão Logística, Random Forest e
  LightGBM; avalia em validação e teste. Métricas reportadas como média ± dp.
- Significância: teste t pareado (LightGBM vs. cada baseline) sobre o F1 de
  teste por seed.
- O threshold de decisão é escolhido maximizando F1 na *validação* e fica
  gravado no artefato do modelo (nunca ajustado no teste).
- O LightGBM da seed mediana (F1 de validação) é registrado no MLflow Model
  Registry com alias de produção e exportado para ``artifacts/model``.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_mlops.config import PROJECT_ROOT, load_config, resolve
from churn_mlops.data.preprocess import load_processed
from churn_mlops.models.wrapper import ChurnModel


def setup_mlflow(cfg: dict) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{PROJECT_ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "threshold": threshold,
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold que maximiza F1 na validação (grade de percentis das probas)."""
    grid = np.unique(np.quantile(proba, np.linspace(0.05, 0.95, 91)))
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def make_preprocessor(schema: dict) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), schema["numeric_features"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                schema["categorical_features"],
            ),
        ]
    )


def make_baseline(name: str, schema: dict, seed: int):
    prep = make_preprocessor(schema)
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(name)
    return Pipeline([("prep", prep), ("clf", clf)])


def make_lgbm(cfg: dict, seed: int) -> LGBMClassifier:
    return LGBMClassifier(**cfg["model"]["lightgbm"], random_state=seed, n_jobs=-1, verbose=-1)


def fit_lgbm(cfg: dict, schema: dict, train_df: pd.DataFrame, seed: int, target: str) -> ChurnModel:
    """Treina um LightGBM e devolve o wrapper pronto para inferência.

    Também usada pelo retreinamento automático da simulação, para que produção
    e retreino compartilhem exatamente o mesmo procedimento.
    """
    model = ChurnModel(estimator=make_lgbm(cfg, seed), schema=schema, trained_at=ChurnModel.now())
    X = model.prepare(train_df)
    model.estimator.fit(X, train_df[target])
    return model


def _log_run(name: str, seed: int, params: dict, metrics_val: dict, metrics_test: dict) -> None:
    with mlflow.start_run(run_name=f"{name}-seed{seed}", nested=True):
        mlflow.log_params({"model": name, "seed": seed, **params})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics_val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items()})


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    target = cfg["data"]["target"]
    parts, schema = load_processed(cfg)
    train_df, val_df, test_df = parts["train"], parts["val"], parts["test"]
    y_val, y_test = val_df[target], test_df[target]

    setup_mlflow(cfg)
    seeds = [cfg["seed"] + i for i in range(cfg["n_seeds"])]
    results: dict[str, list[dict]] = {}
    lgbm_models: list[tuple[float, ChurnModel]] = []

    with mlflow.start_run(run_name="training-campaign"):
        for seed in seeds:
            # -------- baselines sklearn --------
            for name in ("logistic_regression", "random_forest"):
                pipe = make_baseline(name, schema, seed)
                Xtr = train_df[schema["numeric_features"] + schema["categorical_features"]]
                pipe.fit(Xtr, train_df[target])
                proba_val = pipe.predict_proba(val_df[Xtr.columns])[:, 1]
                thr = best_f1_threshold(y_val, proba_val)
                m_val = compute_metrics(y_val, proba_val, thr)
                m_test = compute_metrics(y_test, pipe.predict_proba(test_df[Xtr.columns])[:, 1], thr)
                results.setdefault(name, []).append(m_test)
                _log_run(name, seed, {}, m_val, m_test)

            # -------- modelo principal --------
            model = fit_lgbm(cfg, schema, train_df, seed, target)
            proba_val = model.predict_proba(val_df)
            model.threshold = best_f1_threshold(y_val, proba_val)
            m_val = compute_metrics(y_val, proba_val, model.threshold)
            m_test = compute_metrics(y_test, model.predict_proba(test_df), model.threshold)
            model.metrics = {"val": m_val, "test": m_test}
            results.setdefault("lightgbm", []).append(m_test)
            lgbm_models.append((m_val["f1"], model))
            _log_run("lightgbm", seed, cfg["model"]["lightgbm"], m_val, m_test)

        # -------- agregação e significância --------
        summary = {
            name: {
                k: {
                    "mean": float(np.mean([r[k] for r in runs])),
                    "std": float(np.std([r[k] for r in runs])),
                }
                for k in runs[0]
            }
            for name, runs in results.items()
        }
        ttests = {}
        lgbm_f1 = [r["f1"] for r in results["lightgbm"]]
        for base in ("logistic_regression", "random_forest"):
            t, p = stats.ttest_rel(lgbm_f1, [r["f1"] for r in results[base]])
            ttests[f"lightgbm_vs_{base}"] = {"t": float(t), "p_value": float(p)}
        mlflow.log_metrics(
            {f"{name}_test_f1_mean": s["f1"]["mean"] for name, s in summary.items()}
        )

        # -------- seleção e registro do modelo de produção --------
        lgbm_models.sort(key=lambda pair: pair[0])
        production = lgbm_models[len(lgbm_models) // 2][1]  # seed mediana em F1 de validação
        reg_name = cfg["mlflow"]["registered_model"]
        with mlflow.start_run(run_name="production-model", nested=True):
            mlflow.log_params({"model": "lightgbm", **cfg["model"]["lightgbm"]})
            mlflow.log_metrics({f"test_{k}": v for k, v in production.metrics["test"].items()})
            info = mlflow.lightgbm.log_model(
                production.estimator,
                artifact_path="model",
                registered_model_name=reg_name,
            )
        version = getattr(info, "registered_model_version", None)
        if version is not None:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(reg_name, cfg["mlflow"]["production_alias"], version)
            production.version = str(version)

    model_dir = resolve(cfg["artifacts"]["model_dir"])
    production.save(model_dir)
    fi = pd.Series(production.estimator.feature_importances_, index=production.feature_names)
    fi.sort_values(ascending=False).to_csv(model_dir / "feature_importance.csv", header=["importance"])

    out = {"summary": summary, "paired_ttests": ttests, "production_version": production.version}
    report_dir = resolve(cfg["reports_dir"]) / "training"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name, s in summary.items():
        print(
            f"[train] {name:22s} test F1={s['f1']['mean']:.3f}±{s['f1']['std']:.3f} "
            f"AUC={s['roc_auc']['mean']:.3f}±{s['roc_auc']['std']:.3f} "
            f"PR-AUC={s['pr_auc']['mean']:.3f}"
        )
    for cmp_name, r in ttests.items():
        print(f"[train] {cmp_name}: t={r['t']:.2f} p={r['p_value']:.4f}")
    print(f"[train] modelo de produção: versão MLflow {production.version} -> {model_dir}")
    return out


if __name__ == "__main__":
    run()
