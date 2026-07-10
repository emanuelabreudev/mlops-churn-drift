"""Simulação do ciclo de produção: predizer → monitorar → decidir → retreinar.

Roda o mesmo fluxo de 30 dias em dois modos e compara:

- ``static``: baseline do briefing — modelo treinado uma vez, sem reação
  (o monitoramento roda apenas para registrar a degradação).
- ``auto``: pipeline completo — quando o motor de decisão sinaliza CRITICAL,
  retreina com a janela recente de lotes rotulados (rótulos chegam com
  atraso de 1 dia simulado), registra a nova versão no MLflow Model Registry
  (alias de produção) e redefine a referência de drift.

Saídas em reports/simulation/: monitoring_log_{modo}.csv, psi_log_{modo}.csv
(PSI por feature/dia, usado no dashboard), summary.json (latência de detecção
vs. meta de 3 dias, métricas pré/pós-drift por modo) e relatórios HTML do
Evidently nos dias de mudança de severidade.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn_mlops.config import load_config, resolve
from churn_mlops.data.preprocess import load_processed
from churn_mlops.models.train import best_f1_threshold, fit_lgbm, setup_mlflow
from churn_mlops.models.wrapper import ChurnModel
from churn_mlops.monitoring.evidently_report import save_drift_report
from churn_mlops.monitoring.metrics import evaluate_drift
from churn_mlops.pipeline.retrain import register_retrained

MAX_HTML_REPORTS = 6


def _batch_metrics(model: ChurnModel, batch: pd.DataFrame, target: str) -> dict:
    y = batch[target].to_numpy()
    proba = model.predict_proba(batch)
    pred = (proba >= model.threshold).astype(int)
    both_classes = len(np.unique(y)) > 1
    return {
        "f1": f1_score(y, pred) if both_classes else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred) if both_classes else np.nan,
        "roc_auc": roc_auc_score(y, proba) if both_classes else np.nan,
        # Brier: própria e sensível à calibração — F1 sobe com a prevalência
        # pós-drift e mascara a degradação; AUC e Brier a expõem.
        "brier": brier_score_loss(y, proba),
        "batch_churn_rate": float(y.mean()),
        "mean_proba": float(proba.mean()),
    }


def run_mode(mode: str, cfg: dict, parts: dict, schema: dict, manifest: dict) -> dict:
    target = cfg["data"]["target"]
    rcfg = cfg["simulation"]["retrain"]
    feature_cols = schema["numeric_features"] + schema["categorical_features"]
    out_dir = resolve(cfg["reports_dir"]) / "simulation"

    model = ChurnModel.load(resolve(cfg["artifacts"]["model_dir"]))
    reference = parts["train"][feature_cols]
    model_version = model.version or "1"

    rows, psi_rows, retrain_events = [], [], []
    labeled_history: list[tuple[int, pd.DataFrame]] = []
    last_retrain_day = -(10**9)
    episode_start: int | None = None  # 1º dia CRITICAL do episódio de drift corrente
    prev_severity, html_saved = "OK", 0

    for entry in manifest["days"]:
        day = entry["day"]
        batch = pd.read_parquet(resolve(entry["path"]))

        perf = _batch_metrics(model, batch, target)
        report = evaluate_drift(reference, batch[feature_cols], schema, cfg["drift"])
        for f in report.features:
            psi_rows.append({"mode": mode, "day": day, "feature": f.feature, "psi": f.psi, "status": f.status})

        if report.dataset_drift and episode_start is None:
            episode_start = day

        # Rótulos chegam com 1 dia de atraso: o retreino só acontece quando há
        # lotes rotulados suficientes *do episódio de drift* (senão treinaríamos
        # de novo na distribuição antiga e o modelo não aprenderia o regime novo).
        episode_pool = [b for d, b in labeled_history if episode_start is not None and d >= episode_start]
        retrained = False
        can_retrain = (
            mode == "auto"
            and rcfg["enabled"]
            and episode_start is not None
            and day - last_retrain_day >= rcfg["cooldown_days"]
            and len(episode_pool) >= rcfg["min_batches"]
        )
        if can_retrain:
            pool = pd.concat(episode_pool[-15:], ignore_index=True)
            fit_pool, thr_pool = train_test_split(
                pool, test_size=0.2, stratify=pool[target], random_state=cfg["seed"]
            )
            new_train = pd.concat([parts["train"], fit_pool], ignore_index=True)
            model = fit_lgbm(cfg, schema, new_train, cfg["seed"], target)
            model.threshold = best_f1_threshold(thr_pool[target], model.predict_proba(thr_pool))
            model_version = register_retrained(cfg, model, day, len(new_train))
            model.version = model_version
            # A referência de monitoramento passa a ser o regime ao qual o modelo
            # acabou de se adaptar: a pergunta do monitor vira "a distribuição
            # mudou DE NOVO?" — evita alertas perpétuos sobre o drift já tratado.
            reference = pd.concat(episode_pool, ignore_index=True)[feature_cols]
            episode_start = None
            last_retrain_day, retrained = day, True
            retrain_events.append({"day": day, "version": model_version, "n_train": len(new_train)})
            print(f"[simulate:{mode}] dia {day}: CRITICAL -> retreinado (v{model_version}, n={len(new_train)})")

        rows.append(
            {
                "mode": mode, "day": day, "n": len(batch), "drift_injected": entry["drifted"],
                **perf, **report.to_row(), "model_version": model_version, "retrained": retrained,
            }
        )

        if html_saved < MAX_HTML_REPORTS and (
            day == 1 or (report.severity != prev_severity and report.severity != "OK") or retrained
        ):
            tag = "retrain" if retrained else report.severity.lower()
            save_drift_report(reference, batch[feature_cols], out_dir / f"evidently_{mode}_day{day:02d}_{tag}.html")
            html_saved += 1
        prev_severity = report.severity

        labeled_history.append((day, batch))

    log = pd.DataFrame(rows)
    log.to_csv(out_dir / f"monitoring_log_{mode}.csv", index=False)
    pd.DataFrame(psi_rows).to_csv(out_dir / f"psi_log_{mode}.csv", index=False)

    injection = manifest["injection_day"]
    critical_days = log.loc[log["severity"] == "CRITICAL", "day"]
    detected = critical_days[critical_days >= injection]
    detection_day = int(detected.iloc[0]) if len(detected) else None
    first_retrain = retrain_events[0]["day"] if retrain_events else None

    pre = log[log["day"] < injection]
    post = log[log["day"] >= injection]
    result = {
        "mode": mode,
        "detection_day": detection_day,
        "detection_latency_days": None if detection_day is None else detection_day - injection,
        "false_alarms_pre_injection": int((critical_days < injection).sum()),
        "retrain_events": retrain_events,
    }
    for metric in ("f1", "roc_auc", "brier"):
        result[f"{metric}_pre"] = float(pre[metric].mean())
        result[f"{metric}_post"] = float(post[metric].mean())
        if first_retrain is not None:
            result[f"{metric}_post_retrain"] = float(log.loc[log["day"] > first_retrain, metric].mean())
    return result


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    parts, schema = load_processed(cfg)
    out_dir = resolve(cfg["reports_dir"]) / "simulation"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    setup_mlflow(cfg)
    mlflow.set_experiment("churn-simulation")

    results = {mode: run_mode(mode, cfg, parts, schema, manifest) for mode in ("static", "auto")}
    latency = results["auto"]["detection_latency_days"]
    summary = {
        "injection_day": manifest["injection_day"],
        "intensity": manifest["intensity"],
        "goal_detection_days": 3,
        "detection_day": results["auto"]["detection_day"],
        "detection_latency_days": latency,
        "goal_met": latency is not None and latency <= 3,
        "modes": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\n[simulate] injeção dia {manifest['injection_day']} | detecção dia "
        f"{summary['detection_day']} (latência {latency}d, meta <=3d: "
        f"{'ATINGIDA' if summary['goal_met'] else 'NÃO atingida'})"
    )
    for mode, r in results.items():
        extra = (
            f" | AUC pós-retreino={r['roc_auc_post_retrain']:.3f} Brier={r['brier_post_retrain']:.3f}"
            if "roc_auc_post_retrain" in r
            else ""
        )
        print(
            f"[simulate] {mode:6s}: AUC pré={r['roc_auc_pre']:.3f} pós={r['roc_auc_post']:.3f} | "
            f"Brier pré={r['brier_pre']:.3f} pós={r['brier_post']:.3f}{extra}"
        )
    return summary


if __name__ == "__main__":
    run()
