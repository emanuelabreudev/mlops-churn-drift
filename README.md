# Pipeline de MLOps com Monitoramento de Data Drift

Pipeline completo de machine learning em produção (simulada) para
**classificação de churn de clientes**: treinamento versionado, serving via
API containerizada, monitoramento contínuo de drift e **retreinamento
automático** quando a distribuição dos dados muda.

> Opção de projeto 9 — Machine Learning / Engenharia de Dados / Técnicas
> Computacionais. Proposta revisada em [`docs/proposta.md`](docs/proposta.md);
> resultados canônicos em [`docs/resultados.md`](docs/resultados.md);
> decisões de stack em [`docs/adr/`](docs/adr/).

## Resultados em uma linha

**O drift injetado no dia 15 foi detectado no próprio dia 15 (latência 0,
meta ≤ 3 dias), com zero falsos alarmes em 14 dias de controle**; o
retreinamento automático (dia 19, aguardando rótulos do episódio) recuperou a
calibração do modelo (Brier melhor que o estático em 10/11 dias), silenciou
os alertas e registrou a nova versão no MLflow Model Registry.

| | AUC teste | F1 teste | Detecção | Falsos alarmes |
|---|---|---|---|---|
| LightGBM (produção) | **0,856 ± 0,000** | **0,639 ± 0,001** | — | — |
| vs. melhor baseline | 0,849 (p < 0,001) | 0,630 | — | — |
| Pipeline de drift | — | — | **latência 0 d** (meta ≤ 3) | **0** |

## Arquitetura

```mermaid
flowchart LR
    subgraph Treino
        A[Telco Churn CSV<br/>SHA-256 verificado] --> B[preprocess<br/>70/15/15 estratificado]
        B --> C[LightGBM + baselines<br/>5 seeds, teste t pareado]
        C --> D[(MLflow<br/>Tracking + Registry<br/>alias @production)]
        C --> E[artifacts/model<br/>wrapper: schema + threshold]
    end
    subgraph Produção simulada — 30 dias
        F[Copula gaussiana<br/>+ cenário de drift D15] -->|lote diário| G[API FastAPI<br/>/predict /metrics /reload]
        E --> G
        G --> H[Monitor: PSI + KS + χ²<br/>por feature/lote]
        H -->|CRITICAL + 4 lotes rotulados| I[Retreino automático<br/>nova versão no registry<br/>referência atualizada]
        I --> G
        H --> J[Evidently AI<br/>relatórios HTML]
        H --> K[Dashboard Streamlit<br/>PSI, AUC/Brier, alertas]
    end
```

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) (Python 3.11+ gerenciado
automaticamente). Docker opcional.

```bash
make all        # setup -> download (com checksum) -> EDA -> treino -> drift -> simulação
make test       # 24 testes unitários/integração (dados sintéticos, roda offline)
make dashboard  # dashboard Streamlit em http://localhost:8501
make mlflow-ui  # experimentos e registry em http://localhost:5000
make api        # API em http://localhost:8000/docs
make ablation   # estudo de ablação (intensidade x latência)  [opcional]
docker compose up --build   # stack completa: mlflow + api + dashboard
```

Exemplo de predição:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "customers": [{"gender":"Female","SeniorCitizen":"No","Partner":"Yes",
  "Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No",
  "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
  "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
  "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check","MonthlyCharges":89.9,"TotalCharges":179.8}]}'
# -> {"predictions":[{"churn_probability":0.773,"churn_predicted":true}],"model_version":"1",...}
```

## 1. Problema

Modelos de churn são avaliados no treino e esquecidos em produção. Quando a
distribuição muda (campanha comercial, novo portfólio), a degradação só é
percebida quando a receita já foi perdida. Monitoramento contínuo é o débito
técnico mais caro de sistemas de ML (Sculley et al., 2015). Este projeto
trata detecção e reação a drift como **parte central da arquitetura**, não
como extensão: hipóteses explícitas, meta mensurável (detecção ≤ 3 dias
simulados) e comparação controlada contra um modelo estático.

**Stakeholders simulados:** time de retenção (consome as predições via API),
time de plataforma (opera serving + monitoramento), equipe de DS (dona do
modelo e dos limiares).

## 2. Dados

[Data card completo](data/DATASET_CARD.md) — Telco Customer Churn (Kaggle,
CC0), 7.043 × 21, churn 26,5%, SHA-256 fixado e verificado a cada download.
EDA reproduzível (`make eda` → `reports/eda/`): tratamento documentado de
`TotalCharges` (11 blanks ⇔ `tenure == 0`), churn de 42,7% em contratos
month-to-month vs. 2,8% em bienais, colinearidade tenure×TotalCharges.

**Produção simulada:** copula gaussiana (marginais empíricas + correlação no
espaço normal — mesmo princípio do `GaussianCopulaSynthesizer` do SDV, ver
[ADR-0003](docs/adr/0003-copula-interna-vs-sdv.md)) gera 30 lotes diários de
250 clientes. A partir do dia 15, cenário **"campanha comercial"**: clientes
mais novos, +month-to-month, +fibra, mensalidades +25% e odds de churn ×1,6.
Rótulos vêm de um oráculo calibrado (RF + isotônica) — P(Y|X) coerente entre
regimes, permitindo medir degradação e recuperação honestamente. O dia da
injeção fica registrado como ground truth. SDV real disponível via
`uv sync --extra sdv` + `synthesizer: sdv` no config.

## 3. Metodologia

- **Protocolo:** split 70/15/15 estratificado; teste intocado; threshold de
  decisão tunado só na validação e versionado no artefato; 5 seeds com
  média ± dp e teste t pareado.
- **Baselines:** modelo estático sem monitoramento (grupo de controle do
  briefing), Regressão Logística e Random Forest.
- **Modelo:** LightGBM raso e regularizado, **sem class weights** — a
  calibração importa para o gatilho e para o negócio
  ([ADR-0001](docs/adr/0001-modelo-lightgbm.md)).
- **Tracking:** MLflow + Model Registry com alias `@production`; cada
  retreino vira nova versão ([ADR-0002](docs/adr/0002-mlflow-tracking-registry.md)).
- **Monitoramento:** motor de decisão próprio — PSI (limiares de Siddiqi,
  2017: ≥ 0,25 crítico, 0,10–0,25 moderado), KS e χ² por feature; drift de
  dataset ⇔ alguma feature crítica ou ≥ 50% moderadas. Evidently AI gera os
  relatórios visuais; decisão e visualização desacopladas
  ([ADR-0004](docs/adr/0004-evidently-e-motor-de-decisao-proprio.md)).
- **Retreinamento automático:** política de episódio — rótulos chegam com 1
  dia de atraso; o gatilho espera ≥ 4 lotes rotulados **do episódio de
  drift** (retreinar antes disso significaria treinar na distribuição
  antiga), respeita cooldown, re-tuna o threshold no regime novo e atualiza
  a referência de monitoramento (a pergunta vira "mudou *de novo*?" —
  elimina fadiga de alerta).
- **Serving:** FastAPI + Docker com validação Pydantic, `/metrics`
  Prometheus e `/reload` sem downtime
  ([ADR-0005](docs/adr/0005-fastapi-streamlit-orquestracao.md)).

## 4. Resultados

Números completos e leitura crítica em [`docs/resultados.md`](docs/resultados.md).
Destaques:

1. **Detecção:** latência 0 dias em todas as intensidades da ablação
   (light/moderate/severe), zero falsos alarmes — meta de ≤ 3 dias atingida
   com folga.
2. **Degradação real do estático:** AUC −3,8 p.p. e Brier +43% pós-drift.
   O F1 *sobe* pós-drift (taxa base 24% → 44%) — armadilha de métrica
   documentada: monitorar F1 mascararia o problema.
3. **Recuperação pelo retreino:** calibração restaurada (proba média 0,41 vs
   taxa real 0,45; estático fica em 0,36), Brier melhor que o estático em
   10/11 dias, F1 +0,7 p.p., alertas zerados após adaptação. A AUC não
   volta ao nível pré-drift: o regime novo é intrinsecamente menos separável
   — retreino não recria sinal que deixou de existir (achado discutido em
   `docs/resultados.md` §3).
4. **H1:** LightGBM supera os baselines com significância (p < 0,001 e
   p = 0,013) — mas só após tuning; a config default perdia para a Regressão
   Logística (lição registrada no ADR-0001).

## 5. Reprodutibilidade

- Ambiente: `uv` com `uv.lock` commitado; Python gerenciado; Docker para a
  stack de serving.
- Dados: download com SHA-256 verificado; pré-processamento 100% script;
  seeds fixas em todos os estágios (copula, cenário, treino, splits).
- CI (GitHub Actions): lint (ruff) + 24 testes + smoke test do pipeline
  ponta a ponta com dados sintéticos — roda offline, sem o dataset.
- Um comando: `make all` regenera tudo (dados → EDA → treino → simulação).

## Estrutura do repositório

```
├── configs/config.yaml          # único ponto de configuração (limiares, cenário, modelo)
├── src/churn_mlops/
│   ├── data/                    # download c/ checksum, preprocess, schema
│   ├── models/                  # wrapper (schema+threshold), treino multi-seed
│   ├── drift/                   # copula interna, SDV opcional, cenário c/ ground truth
│   ├── monitoring/              # PSI/KS/χ² + decisão; relatórios Evidently
│   ├── pipeline/                # simulação 30 dias, retreino, ablação
│   ├── serving/                 # FastAPI (predict, metrics, reload)
│   └── eda.py
├── dashboard/app.py             # Streamlit: PSI, AUC/Brier, heatmap, alertas
├── tests/                       # 24 testes, dados sintéticos, offline
├── docker/ + docker-compose.yml # mlflow + api + dashboard
├── docs/{proposta,resultados}.md + docs/adr/000{1..5}*.md
└── data/DATASET_CARD.md
```

## Trabalho futuro

- Estimação de performance sem rótulos (NannyML/CBPE) para detectar concept
  drift antes de os rótulos chegarem.
- Streaming (Kafka/Flink) com PSI/KS em janelas — o motor de decisão migra
  sem mudança (tendência 2026 citada no briefing).
- Fairness: monitorar drift e erro por subgrupo (gênero/senioridade).
- Explicabilidade (SHAP) nos alertas: *quais clientes* mudaram, não só quais
  features.

## Referências

Gama et al. (2014), *ACM Computing Surveys* 46(4) · Lu et al. (2018), *IEEE
TKDE* 31(12) · Patki et al. (2016), *IEEE DSAA* · Sculley et al. (2015),
*NeurIPS 28* · Siddiqi (2017), *Intelligent Credit Scoring*, Wiley · Xu et
al. (2019), *NeurIPS 32* · Docs: [Evidently](https://docs.evidentlyai.com) ·
[SDV](https://docs.sdv.dev) · [MLflow](https://mlflow.org) ·
[LightGBM](https://lightgbm.readthedocs.io)

## Licença

MIT (código). Dataset: CC0 (domínio público).
