# Especificação Formal de Projeto — Pipeline de MLOps com Monitoramento de Data Drift

**Predição de churn de clientes com retreinamento automático**
Opção de projeto 9 — Machine Learning / Engenharia de Dados / Técnicas Computacionais
Versão 2.0 (revisada após implementação) — Julho de 2026

> Esta é a revisão da proposta original incorporando as decisões tomadas
> durante a implementação. As mudanças em relação à v1.0 estão marcadas com
> **[revisado]** e justificadas nos ADRs (`docs/adr/`). Os resultados
> canônicos estão em `docs/resultados.md` e no README.

## Aderência aos critérios de avaliação

| Critério (peso) | Seção nesta proposta | Evidência no repositório |
|---|---|---|
| 2.1 Definição do Problema (0,15) | §1 | README §Problema |
| 2.2 Dados e Pré-processamento (0,15) | §2 | `data/DATASET_CARD.md`, `reports/eda/`, `src/churn_mlops/data/` |
| 2.3 Metodologia (0,20) | §3–4 | `src/churn_mlops/`, `configs/config.yaml`, ADRs |
| 2.4 Resultados e Métricas (0,20) | §5 | `docs/resultados.md`, `reports/` |
| 2.5 Documentação / Reprodutibilidade (0,15) | §7 | README, `Makefile`, CI, `uv.lock` |

## 1. Problema e motivação

Modelos de churn são tipicamente avaliados só no momento do treino. Quando a
distribuição dos dados de produção muda — campanha comercial, novo portfólio,
sazonalidade — o modelo passa a operar sobre premissas velhas e a degradação
só é notada quando a receita já foi perdida. Sculley et al. (2015) apontam o
monitoramento contínuo como o débito técnico mais negligenciado em sistemas
de ML; Gama et al. (2014) e Lu et al. (2018) sistematizam a adaptação a
*concept drift* pressupondo exatamente o mecanismo que este projeto
implementa: detecção contínua + reação automática.

**Problema central:** projetar e implementar um pipeline de produção para
classificação de churn que seja (i) versionado e reprodutível, (ii) servido
via API, (iii) monitorado continuamente quanto a drift e (iv) capaz de
retreinar automaticamente quando a mudança ultrapassar limiar predefinido.

**Objetivos específicos:**

- OE1 — LightGBM com AUC-ROC ≥ 0,82 e F1 (churn) ≥ 0,55 no teste, superando
  baselines com significância estatística.
- OE2 — Serving REST (FastAPI) containerizado, com validação de payload,
  healthcheck, métricas Prometheus e reload sem downtime.
- OE3 — Monitoramento por lote com PSI + KS (numéricas) + qui-quadrado
  (categóricas), relatórios visuais Evidently.
- OE4 — Detectar o drift injetado em ≤ 3 dias simulados (meta do briefing).
- OE5 — Retreinamento automático quando PSI ≥ 0,25 em alguma feature ou ≥ 50%
  das features com PSI ≥ 0,10, com política explícita de rótulos atrasados,
  cooldown e mínimo de lotes do episódio de drift. **[revisado]**
- OE6 — Dashboard Streamlit com métricas de drift/performance e marcadores
  de injeção/detecção/retreino; ADRs para cada escolha de stack.

**Hipóteses:**

- H1 — LightGBM supera os baselines em F1/AUC no teste (t pareado, α=0,05).
- H2 — O drift sintético degrada de forma mensurável o modelo estático em
  métricas insensíveis à prevalência (AUC ↓, Brier ↑). **[revisado: F1 é
  confundido pelo aumento da taxa base pós-drift — ver resultados]**
- H3 — O pipeline detecta o drift em ≤ 3 dias simulados.
- H4 — O retreinamento automático recupera calibração (Brier) e silencia os
  alertas, restaurando o desempenho ao teto permitido pelo novo regime.
  **[revisado: o teto de AUC do novo regime pode ser menor que o antigo —
  hipótese refinada após os primeiros experimentos]**

## 2. Dados

Telco Customer Churn (Kaggle, CC0), 7.043 × 21, churn 26,5%. Data card
completo com origem, mirror, SHA-256 verificado em runtime, schema, vieses e
tratamento em `data/DATASET_CARD.md`. EDA reproduzível (`make eda`).

Drift sintético: copula gaussiana ajustada no treino gera lotes diários
(250 clientes/dia × 30 dias); a partir do dia 15, cenário "campanha
comercial" (clientes mais novos, mais month-to-month/fibra, mensalidades
+25%, odds de churn ×1,6). Rótulos vêm de um oráculo calibrado (RF +
isotônica) que faz o papel do processo gerador do mundo real.
**[revisado: SDV vira engine opcional (`--extra sdv`); a copula interna
evita ~2 GB de dependências PyTorch — ADR-0003]**

## 3. Metodologia

- Split estratificado 70/15/15; teste intocado até a avaliação final.
- Baselines: modelo estático sem monitoramento (grupo de controle do
  briefing), Regressão Logística e Random Forest.
- Modelo principal: LightGBM raso e regularizado (varredura na validação),
  **sem** class weights — calibração preservada; ponto de operação via
  threshold tunado na validação e versionado no artefato. **[revisado]**
- 5 seeds, média ± dp, teste t pareado vs. baselines.
- MLflow (SQLite) para tracking; Model Registry com alias `@production`;
  cada retreino gera nova versão.
- Monitoramento: motor de decisão próprio (PSI/KS/χ², limiares Siddiqi 2017)
  + relatórios Evidently — decisão desacoplada de visualização (ADR-0004).
- Retreinamento: política de episódio — espera ≥ 4 lotes rotulados do
  episódio de drift (rótulos chegam com 1 dia de atraso), retreina com
  treino original + janela recente, re-tuna o threshold no regime novo e
  **redefine a referência de monitoramento para o regime novo** (a pergunta
  do monitor vira "mudou de novo?", eliminando fadiga de alerta).
  **[revisado — decisão de design central, ver README]**
- Ablação: intensidade do drift (light/moderate/severe) × latência de
  detecção, falsos alarmes e degradação.

## 4. Cronograma executado

| Fase | Entregável |
|---|---|
| Setup + dados + EDA | data card, relatório EDA |
| Baselines + modelo principal | métricas multi-seed no MLflow |
| Serving | API FastAPI + Docker + testes |
| Drift sintético | gerador de cenário com ground truth |
| Monitoramento + gatilho | motor PSI/KS/χ² testado + Evidently |
| Simulação 30 dias | comparação estático × auto |
| Ablação + dashboard + docs | summary.json, Streamlit, ADRs, README |

## 5. Resultados esperados vs. obtidos

Ver `docs/resultados.md` (números canônicos da execução com seed 42) e
README §Resultados: metas OE1–OE6 atingidas; H1 e H3 confirmadas; H2
confirmada em AUC/Brier (com a ressalva do F1); H4 confirmada para
calibração e alertas, refinada para ranking (AUC).

## 6. Limitações

- Operadora fictícia; sem generalização direta para outros mercados.
- Drift sintético (mesmo via copula/SDV) não captura toda a complexidade de
  drifts reais fora da distribuição aprendida.
- Sem rótulos em tempo real, o monitoramento detecta mudança em P(X); o
  deslocamento de conceito é injetado e avaliado, mas a *detecção* de
  P(Y|X) exigiria rótulos atrasados sistemáticos (candidato: NannyML).
- Streaming Kafka/Flink (tendência 2026) fica como trabalho futuro; o motor
  de decisão migra sem mudança.

## Referências

Gama, J. et al. (2014). A survey on concept drift adaptation. *ACM Computing
Surveys*, 46(4). · Lu, J. et al. (2018). Learning under concept drift: a
review. *IEEE TKDE*, 31(12). · Patki, N. et al. (2016). The Synthetic Data
Vault. *IEEE DSAA*. · Sculley, D. et al. (2015). Hidden technical debt in
machine learning systems. *NeurIPS 28*. · Siddiqi, N. (2017). *Intelligent
Credit Scoring* (2ª ed.). Wiley. · Xu, L. et al. (2019). Modeling tabular
data using conditional GAN. *NeurIPS 32*.
