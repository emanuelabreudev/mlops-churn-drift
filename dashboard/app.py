"""Dashboard de monitoramento (Streamlit) — entregável diferenciador.

Lê os logs produzidos pela simulação (reports/simulation/) e exibe:
- linha do tempo de PSI com limiares e marcadores de injeção/detecção/retreino;
- comparação de performance (F1/AUC) entre o modelo estático e o pipeline
  com retreinamento automático;
- heatmap de PSI por feature/dia e tabela de alertas.

Rode com: make dashboard  (ou: uv run streamlit run dashboard/app.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "reports" / "simulation"

st.set_page_config(page_title="Churn MLOps — Drift Monitor", page_icon="📡", layout="wide")
st.title("📡 Monitoramento de Data Drift — Churn MLOps")

if not (SIM_DIR / "summary.json").exists():
    st.warning("Nenhuma simulação encontrada. Rode `make simulate` antes de abrir o dashboard.")
    st.stop()

summary = json.loads((SIM_DIR / "summary.json").read_text(encoding="utf-8"))
logs = {m: pd.read_csv(SIM_DIR / f"monitoring_log_{m}.csv") for m in ("static", "auto")}
psi = pd.read_csv(SIM_DIR / "psi_log_auto.csv")

injection = summary["injection_day"]
detection = summary["detection_day"]
latency = summary["detection_latency_days"]
retrains = [e["day"] for e in summary["modes"]["auto"]["retrain_events"]]

# ---------------- KPIs ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dia da injeção (ground truth)", f"D{injection}")
c2.metric("Dia da detecção", f"D{detection}" if detection else "—")
c3.metric(
    "Latência de detecção",
    f"{latency} dia(s)",
    delta=f"meta ≤ 3 — {'atingida ✅' if summary['goal_met'] else 'não atingida ❌'}",
    delta_color="off",
)
c4.metric("Retreinos automáticos", len(retrains))
auto, static = summary["modes"]["auto"], summary["modes"]["static"]
rec = auto.get("roc_auc_post_retrain", auto["roc_auc_post"])
c5.metric(
    "AUC pós-drift (auto vs estático)",
    f"{rec:.3f}",
    delta=f"{(rec - static['roc_auc_post']):+.3f} vs estático",
)

st.divider()


def add_markers(fig: go.Figure) -> go.Figure:
    fig.add_vline(x=injection, line_dash="dash", line_color="#E45756")
    fig.add_annotation(x=injection, yref="paper", y=1.06, text="injeção", showarrow=False,
                       font={"color": "#E45756"})
    if detection:
        fig.add_vline(x=detection, line_dash="dot", line_color="#F58518")
        fig.add_annotation(x=detection, yref="paper", y=0.98, text="detecção", showarrow=False,
                           font={"color": "#F58518"})
    for d in retrains:
        fig.add_vline(x=d, line_dash="dashdot", line_color="#54A24B")
        fig.add_annotation(x=d, yref="paper", y=1.06, text="retreino", showarrow=False,
                           font={"color": "#54A24B"})
    return fig


# ---------------- Drift ----------------
left, right = st.columns(2)
with left:
    st.subheader("PSI agregado por dia (pipeline auto)")
    log = logs["auto"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=log["day"], y=log["max_psi"], name="PSI máximo", line={"color": "#4C78A8"}))
    fig.add_trace(go.Scatter(x=log["day"], y=log["mean_psi"], name="PSI médio",
                             line={"color": "#72B7B2", "dash": "dot"}))
    fig.add_hline(y=0.25, line_color="#E45756", line_dash="dash",
                  annotation_text="crítico (0.25)", annotation_position="top left")
    fig.add_hline(y=0.10, line_color="#F58518", line_dash="dash",
                  annotation_text="moderado (0.10)", annotation_position="bottom left")
    add_markers(fig)
    fig.update_layout(xaxis_title="dia simulado", yaxis_title="PSI", height=380,
                      legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fração de features em drift")
    fig = go.Figure()
    for mode, color in [("auto", "#4C78A8"), ("static", "#B0B0B0")]:
        fig.add_trace(go.Scatter(x=logs[mode]["day"], y=logs[mode]["share_drifted"],
                                 name=mode, line={"color": color}))
    fig.add_hline(y=0.5, line_color="#E45756", line_dash="dash", annotation_text="limiar dataset drift")
    add_markers(fig)
    fig.update_layout(xaxis_title="dia simulado", yaxis_title="fração de features (PSI ≥ 0.10)",
                      height=380, legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Performance ----------------
st.subheader("Performance do modelo: estático vs. retreinamento automático")
metric = st.radio(
    "Métrica",
    ["roc_auc", "brier", "f1", "batch_churn_rate", "mean_proba"],
    horizontal=True,
    help="F1 sobe com a prevalência pós-drift; AUC e Brier expõem a degradação real.",
)
fig = go.Figure()
for mode, color in [("static", "#B0B0B0"), ("auto", "#4C78A8")]:
    fig.add_trace(go.Scatter(x=logs[mode]["day"], y=logs[mode][metric], name=mode, line={"color": color}))
add_markers(fig)
fig.update_layout(xaxis_title="dia simulado", yaxis_title=metric, height=400,
                  legend={"orientation": "h", "y": -0.2})
st.plotly_chart(fig, use_container_width=True)

# ---------------- Heatmap por feature ----------------
st.subheader("PSI por feature × dia (pipeline auto)")
pivot = psi.pivot_table(index="feature", columns="day", values="psi")
order = pivot.max(axis=1).sort_values(ascending=False).index
fig = px.imshow(pivot.loc[order], color_continuous_scale="YlOrRd", zmin=0, zmax=0.5, aspect="auto",
                labels={"x": "dia simulado", "y": "", "color": "PSI"})
fig.update_layout(height=520)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Alertas ----------------
st.subheader("Alertas")
alerts = logs["auto"].query("severity != 'OK'")[
    ["day", "severity", "share_drifted", "max_psi", "model_version", "retrained"]
]
st.dataframe(alerts, use_container_width=True, hide_index=True)

st.caption(
    f"Cenário: {summary['intensity']} | injeção D{injection} | "
    f"relatórios Evidently em reports/simulation/*.html"
)
