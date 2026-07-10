"""Cenário temporal de produção simulada com injeção de drift controlada.

Gera ``n_days`` lotes diários. Até a véspera de ``injection_day`` os lotes
vêm da distribuição de referência (grupo de controle); a partir dele, aplica
o cenário "campanha comercial": onda de clientes novos, mais contratos
month-to-month, mais fibra óptica e mensalidades mais altas (drift de
covariáveis, tipo *súbito* na taxonomia de Lu et al., 2018), combinado com
um deslocamento leve de conceito (odds de churn multiplicadas), como numa
campanha que atrai clientes menos fiéis.

Os rótulos vêm de um "oráculo" — um RandomForest treinado no dataset
completo que faz o papel do processo gerador do mundo real: para cada linha
sintética, sorteia-se Churn ~ Bernoulli(p_oraculo). Isso mantém P(Y|X)
coerente entre os períodos pré e pós-drift (a menos do deslocamento de
conceito explícito e documentado), o que permite medir de forma honesta a
degradação do modelo e a recuperação após retreino.

O dia exato da injeção fica registrado no manifest (ground truth) para
medir objetivamente a latência de detecção — meta: <= 3 dias simulados.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn_mlops.config import load_config, resolve
from churn_mlops.data.preprocess import load_processed
from churn_mlops.drift.synthesizer import enforce_structural_consistency, make_synthesizer
from churn_mlops.models.train import make_preprocessor

INTENSITIES = {
    "light": {"charge_mult": 1.10, "contract_p": 0.20, "fiber_p": 0.15, "tenure_factor": 0.85, "odds": 1.2},
    "moderate": {"charge_mult": 1.25, "contract_p": 0.45, "fiber_p": 0.35, "tenure_factor": 0.65, "odds": 1.6},
    "severe": {"charge_mult": 1.45, "contract_p": 0.70, "fiber_p": 0.55, "tenure_factor": 0.45, "odds": 2.2},
}

def apply_covariate_drift(batch: pd.DataFrame, params: dict, rng: np.random.Generator) -> pd.DataFrame:
    out = batch.copy()
    n = len(out)

    out["MonthlyCharges"] = out["MonthlyCharges"] * params["charge_mult"]
    out["tenure"] = (out["tenure"] * params["tenure_factor"]).round().clip(lower=0)

    flip_contract = rng.random(n) < params["contract_p"]
    out.loc[flip_contract, "Contract"] = "Month-to-month"

    flip_fiber = rng.random(n) < params["fiber_p"]
    out.loc[flip_fiber, "InternetService"] = "Fiber optic"

    flip_pay = rng.random(n) < params["fiber_p"]
    out.loc[flip_pay, "PaymentMethod"] = "Electronic check"

    # coerência interna: acumulado ~ mensalidade x tempo
    noise = rng.uniform(0.9, 1.05, n)
    out["TotalCharges"] = (out["MonthlyCharges"] * out["tenure"].clip(lower=0) * noise).round(2)
    # linhas que ganharam fibra precisam de add-ons coerentes ("No", não "No internet service")
    return enforce_structural_consistency(out)


def shift_odds(p: np.ndarray, k: float) -> np.ndarray:
    """Multiplica as odds de churn por k (deslocamento leve de conceito)."""
    return (p * k) / (1.0 - p + p * k)


def build_oracle(cfg: dict, schema: dict, full_df: pd.DataFrame):
    """Processo gerador de rótulos do 'mundo real' (independente do modelo servido).

    RandomForest *sem* class_weight + calibração isotônica: as probabilidades
    precisam ser calibradas para que Churn ~ Bernoulli(p) reproduza a taxa de
    churn real (~26,5%) no período pré-drift. Um oráculo com pesos balanceados
    inflaria a taxa base e contaminaria o grupo de controle com label shift.
    """
    target = cfg["data"]["target"]
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, random_state=cfg["seed"] + 1000, n_jobs=-1
    )
    oracle = Pipeline(
        [
            ("prep", make_preprocessor(schema)),
            ("clf", CalibratedClassifierCV(rf, method="isotonic", cv=3)),
        ]
    )
    X = full_df[schema["numeric_features"] + schema["categorical_features"]]
    oracle.fit(X, full_df[target])
    return oracle


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    sim = cfg["simulation"]
    target = cfg["data"]["target"]
    parts, schema = load_processed(cfg)
    train_df = parts["train"]
    full_df = pd.concat(parts.values(), ignore_index=True)

    params = INTENSITIES[sim["drift_intensity"]]
    rng = np.random.default_rng(cfg["seed"])

    synth = make_synthesizer(sim["synthesizer"], schema, cfg["seed"]).fit(train_df)
    oracle = build_oracle(cfg, schema, full_df)
    feature_cols = schema["numeric_features"] + schema["categorical_features"]

    out_dir = resolve(cfg["reports_dir"]) / "simulation" / "batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_days": sim["n_days"],
        "batch_size": sim["batch_size"],
        "injection_day": sim["injection_day"],
        "scenario": sim["scenario"],
        "intensity": sim["drift_intensity"],
        "synthesizer": sim["synthesizer"],
        "params": params,
        "days": [],
    }

    for day in range(1, sim["n_days"] + 1):
        batch = synth.sample(sim["batch_size"], seed=cfg["seed"] + day)
        drifted = day >= sim["injection_day"]
        if drifted:
            batch = apply_covariate_drift(batch, params, rng)

        p = oracle.predict_proba(batch[feature_cols])[:, 1]
        if drifted:
            p = shift_odds(p, params["odds"])
        batch[target] = rng.binomial(1, p).astype("int8")

        path = out_dir / f"day_{day:02d}.parquet"
        batch.to_parquet(path, index=False)
        manifest["days"].append(
            {"day": day, "path": str(path.relative_to(resolve('.'))), "drifted": drifted,
             "churn_rate": float(batch[target].mean())}
        )

    manifest_path = out_dir.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pre = np.mean([d["churn_rate"] for d in manifest["days"] if not d["drifted"]])
    post = np.mean([d["churn_rate"] for d in manifest["days"] if d["drifted"]])
    print(
        f"[scenario] {sim['n_days']} dias gerados (injeção no dia {sim['injection_day']}, "
        f"intensidade {sim['drift_intensity']}); churn médio pré={pre:.3f} pós={post:.3f} -> {out_dir}"
    )
    return manifest


if __name__ == "__main__":
    run()
