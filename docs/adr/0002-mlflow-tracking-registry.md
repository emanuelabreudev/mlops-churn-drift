# ADR-0002 — MLflow para tracking e Model Registry

**Status:** aceito · **Data:** 2026-07-10

## Contexto

O projeto exige rastreabilidade de experimentos (parâmetros, métricas,
artefatos, múltiplas seeds) e versionamento de modelo com promoção explícita
a produção — inclusive para as versões criadas pelo retreinamento automático.

## Decisão

MLflow com backend SQLite local (`sqlite:///mlflow.db`) e Model Registry com
*aliases* (`@production`). Cada retreino disparado por drift registra uma
nova versão do modelo `churn-classifier` e move o alias — o histórico do
registry conta a história dos retreinos.

Para o serving, o modelo é **exportado para `artifacts/model/` (joblib)** com
schema de features, threshold e metadados embutidos. A API carrega esse
artefato, não o registry: indisponibilidade do MLflow não pode derrubar a
predição (falha de tracking é degradação, não outage).

## Alternativas consideradas

- **Weights & Biases** — SaaS; exigiria conta/rede e fere a meta de
  reprodutibilidade offline da disciplina.
- **DVC + Git puro** — versiona artefatos, mas não oferece UI de comparação
  de runs nem registry com aliases.
- **Servir via `mlflow models serve`** — acopla o serving ao formato MLflow
  e dificulta o wrapper com coerção de schema + threshold customizado.

## Consequências

- `make mlflow-ui` abre a UI local com todos os experimentos e versões.
- SQLite é suficiente para 1 máquina; em produção real trocaríamos por
  Postgres + artifact store S3 (mudança de 1 linha no config).
