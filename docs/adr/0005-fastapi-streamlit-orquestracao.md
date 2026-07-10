# ADR-0005 — FastAPI para serving, Streamlit para dashboard, orquestração por script

**Status:** aceito · **Data:** 2026-07-10

## Serving: FastAPI (vs. Flask / mlflow serve)

- Validação de payload declarativa via Pydantic: um cliente com campo
  faltante recebe 422 com diagnóstico, nunca chega ao modelo (testado).
- OpenAPI/Swagger automático em `/docs` — contrato vivo para o time de
  retenção consumir.
- `POST /reload` recarrega o artefato do modelo sem derrubar o processo:
  promoção de versão com zero downtime.
- Flask exigiria validação e documentação manuais; `mlflow models serve`
  não comporta o wrapper com coerção de schema + threshold.

## Dashboard: Streamlit (vs. Grafana)

- O dashboard consome DataFrames (logs de monitoramento) e precisa de
  gráficos estatísticos específicos (heatmap de PSI por feature/dia,
  marcadores de injeção/detecção/retreino) — natural em Streamlit+Plotly,
  trabalhoso em Grafana.
- Grafana exigiria um time-series DB (Prometheus/Influx) só para o exercício.
  Deixamos a porta aberta: a API expõe `GET /metrics` em formato Prometheus,
  então plugar Grafana depois é configuração, não código.

## Orquestração: script de simulação (vs. Airflow/Prefect)

- O loop diário (predizer → monitorar → decidir → retreinar) é sequencial e
  síncrono; um DAG Airflow adicionaria scheduler, metadados e webserver sem
  nenhum ganho no cenário simulado — dias simulados não são dias de relógio.
- A lógica de gatilho (episódio de drift, cooldown, mínimo de lotes
  rotulados) está isolada em `pipeline/simulate.py`; migrá-la para um
  operator Airflow em produção real é transplante direto.

## Consequências

- Stack inteira roda em `docker compose up` com 3 serviços (mlflow, api,
  dashboard) e zero dependências de infraestrutura externa.
- Kafka/Flink (tendência 2026 citada no briefing) fica explicitamente como
  trabalho futuro: o desenho por lotes com referência versionada migra para
  janelas de streaming sem mudar o motor de decisão.
