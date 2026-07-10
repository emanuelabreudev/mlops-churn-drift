"""Pré-processamento reprodutível do Telco Customer Churn.

Decisões de tratamento (ver também notebook/relatório de EDA):
- ``TotalCharges`` vem como string e contém 11 valores em branco, todos em
  clientes com ``tenure == 0`` (recém-chegados que ainda não faturaram).
  Convertem-se para 0.0, coerente com a semântica do campo.
- ``SeniorCitizen`` vem como 0/1; é mapeado para "No"/"Yes" para ser tratado
  uniformemente como categórico (inclusive nos testes de drift qui-quadrado).
- ``customerID`` é descartado (identificador, sem valor preditivo).
- Split estratificado 70/15/15 (treino/validação/teste) com seed fixa.

O schema resultante (features numéricas, categóricas e categorias válidas)
é salvo em ``data/processed/schema.json`` e é a única fonte de verdade sobre
tipos de coluna para o restante do pipeline (treino, drift, serving).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_mlops.config import load_config, resolve

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def clean(raw: pd.DataFrame, target: str = "Churn", id_column: str = "customerID") -> pd.DataFrame:
    df = raw.copy()
    df = df.drop(columns=[id_column], errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df[target] = df[target].map({"Yes": 1, "No": 0}).astype("int8")

    for col in df.columns:
        if col in NUMERIC_FEATURES or col == target:
            continue
        df[col] = df[col].astype(str)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype("float64")
    return df


def build_schema(df: pd.DataFrame, target: str) -> dict:
    categorical = [c for c in df.columns if c not in NUMERIC_FEATURES and c != target]
    return {
        "target": target,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": categorical,
        "categories": {c: sorted(df[c].unique().tolist()) for c in categorical},
    }


def split(df: pd.DataFrame, target: str, test_size: float, val_size: float, seed: int):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[target], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def run(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    dcfg = cfg["data"]
    raw = pd.read_csv(resolve(dcfg["raw_path"]))
    df = clean(raw, target=dcfg["target"], id_column=dcfg["id_column"])
    schema = build_schema(df, dcfg["target"])

    train, val, test = split(df, dcfg["target"], dcfg["test_size"], dcfg["val_size"], cfg["seed"])

    out_dir = resolve(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        p = out_dir / f"{name}.parquet"
        part.to_parquet(p, index=False)
        paths[name] = p
    schema_path = out_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["schema"] = schema_path

    churn_rate = df[dcfg["target"]].mean()
    print(
        f"[preprocess] total={len(df)} churn_rate={churn_rate:.3f} "
        f"train={len(train)} val={len(val)} test={len(test)} -> {out_dir}"
    )
    return paths


def load_processed(cfg: dict | None = None):
    """Carrega os splits processados e o schema (uso interno do pipeline)."""
    cfg = cfg or load_config()
    out_dir = resolve(cfg["data"]["processed_dir"])
    parts = {n: pd.read_parquet(out_dir / f"{n}.parquet") for n in ("train", "val", "test")}
    schema = json.loads((out_dir / "schema.json").read_text(encoding="utf-8"))
    return parts, schema


if __name__ == "__main__":
    run()
