"""EDA reprodutível: gera figuras e um relatório markdown em reports/eda/.

Mantida como script (e não apenas notebook) para que a análise seja
reexecutável em CI e em qualquer máquina com `make eda`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churn_mlops.config import load_config, resolve
from churn_mlops.data.preprocess import clean

PALETTE = {"No": "#4C78A8", "Yes": "#E45756"}


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    target = cfg["data"]["target"]
    raw = pd.read_csv(resolve(cfg["data"]["raw_path"]))
    df = clean(raw, target=target, id_column=cfg["data"]["id_column"])

    out = resolve(cfg["reports_dir"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    n_blank_totalcharges = int((raw["TotalCharges"].astype(str).str.strip() == "").sum())
    churn_rate = df[target].mean()

    # 1) alvo
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target].map({0: "No", 1: "Yes"}).value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[i] for i in counts.index])
    ax.set_title(f"Distribuição de churn (taxa = {churn_rate:.1%})")
    ax.set_ylabel("clientes")
    fig.tight_layout()
    fig.savefig(out / "01_target.png", dpi=120)
    plt.close(fig)

    # 2) numéricas por classe
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"], strict=True):
        for label, name in [(0, "No"), (1, "Yes")]:
            df.loc[df[target] == label, col].plot.kde(ax=ax, label=f"churn={name}", color=PALETTE[name])
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Distribuições numéricas por classe")
    fig.tight_layout()
    fig.savefig(out / "02_numeric_by_class.png", dpi=120)
    plt.close(fig)

    # 3) churn por contrato e por internet
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["Contract", "InternetService"], strict=True):
        rate = df.groupby(col)[target].mean().sort_values(ascending=False)
        ax.bar(rate.index, rate.values, color="#E45756")
        ax.axhline(churn_rate, ls="--", c="gray", label="taxa média")
        ax.set_title(f"Taxa de churn por {col}")
        ax.tick_params(axis="x", rotation=15)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_churn_by_segment.png", dpi=120)
    plt.close(fig)

    # 4) correlação entre numéricas
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df[["tenure", "MonthlyCharges", "TotalCharges", target]].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlação (numéricas + alvo)")
    fig.tight_layout()
    fig.savefig(out / "04_correlation.png", dpi=120)
    plt.close(fig)

    contract_rates = df.groupby("Contract")[target].mean()
    md = f"""# EDA — Telco Customer Churn

- Registros: **{len(df)}**; taxa de churn: **{churn_rate:.1%}** (dataset moderadamente desbalanceado
  — justifica F1/PR-AUC como métricas principais, não acurácia).
- `TotalCharges` continha **{n_blank_totalcharges} strings vazias**, todas em clientes com `tenure == 0`;
  convertidas para 0.0 (cliente novo, nada faturado).
- Churn concentra-se em contratos **Month-to-month** ({contract_rates.get("Month-to-month", float("nan")):.1%})
  contra {contract_rates.get("Two year", float("nan")):.1%} em contratos de dois anos — `Contract` é a
  feature categórica mais informativa e por isso é um dos alvos da injeção de drift simulada.
- `TotalCharges` é fortemente correlacionado com `tenure` (r≈{df["TotalCharges"].corr(df["tenure"]):.2f}),
  como esperado (acumulado ≈ mensalidade × tempo). LightGBM lida bem com essa colinearidade.
- Clientes com fibra óptica e cobrança mensal alta apresentam churn acima da média — segmento
  usado no cenário de drift "campanha comercial".

![alvo](01_target.png)
![numericas](02_numeric_by_class.png)
![segmentos](03_churn_by_segment.png)
![correlacao](04_correlation.png)
"""
    (out / "README.md").write_text(md, encoding="utf-8")
    print(f"[eda] figuras e relatório em {out}")


if __name__ == "__main__":
    run()
