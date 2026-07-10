"""Estudo de ablação: intensidade do drift × latência de detecção e degradação.

Para cada intensidade (light/moderate/severe), gera uma linha do tempo
independente e roda o monitoramento em modo estático (sem retreino), medindo:
latência de detecção, falsos alarmes pré-injeção e degradação de AUC/Brier.
Responde à pergunta de sensibilidade: "quão sutil um drift pode ser e ainda
ser detectado dentro da meta de 3 dias?"

Saída: reports/ablation/summary.json + subdiretórios por intensidade.
"""

from __future__ import annotations

import copy
import json

from churn_mlops.config import load_config, resolve
from churn_mlops.data.preprocess import load_processed
from churn_mlops.drift import scenario
from churn_mlops.pipeline.simulate import run_mode

INTENSITIES = ("light", "moderate", "severe")


def run(cfg: dict | None = None) -> dict:
    base_cfg = cfg or load_config()
    results = {}

    for intensity in INTENSITIES:
        cfg_i = copy.deepcopy(base_cfg)
        cfg_i["simulation"]["drift_intensity"] = intensity
        cfg_i["reports_dir"] = f"{base_cfg['reports_dir']}/ablation/{intensity}"
        print(f"[ablation] gerando cenário '{intensity}'…")
        scenario.run(cfg_i)

        parts, schema = load_processed(cfg_i)
        manifest = json.loads(
            (resolve(cfg_i["reports_dir"]) / "simulation" / "manifest.json").read_text(encoding="utf-8")
        )
        r = run_mode("static", cfg_i, parts, schema, manifest)
        results[intensity] = {
            "detection_day": r["detection_day"],
            "detection_latency_days": r["detection_latency_days"],
            "false_alarms_pre_injection": r["false_alarms_pre_injection"],
            "goal_met": r["detection_latency_days"] is not None and r["detection_latency_days"] <= 3,
            "roc_auc_pre": r["roc_auc_pre"],
            "roc_auc_post": r["roc_auc_post"],
            "brier_pre": r["brier_pre"],
            "brier_post": r["brier_post"],
        }

    out = resolve(base_cfg["reports_dir"]) / "ablation" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[ablation] {'intensidade':12s} {'latência':>9s} {'falsos+':>8s} {'ΔAUC':>7s} {'ΔBrier':>7s}")
    for k, v in results.items():
        lat = v["detection_latency_days"]
        print(
            f"[ablation] {k:12s} {str(lat) + 'd' if lat is not None else 'não':>9s} "
            f"{v['false_alarms_pre_injection']:>8d} "
            f"{v['roc_auc_post'] - v['roc_auc_pre']:>+7.3f} {v['brier_post'] - v['brier_pre']:>+7.3f}"
        )
    return results


if __name__ == "__main__":
    run()
