# Resultados canônicos

Execução de referência com `configs/config.yaml` (seed 42, 5 seeds no
treino). Todos os números são regeneráveis com `make all`.

## 1. Modelos (conjunto de teste, média ± dp de 5 seeds)

| Modelo | AUC-ROC | PR-AUC | F1 (churn) | Precision | Recall |
|---|---|---|---|---|---|
| Regressão Logística | 0,849 ± 0,000 | 0,641 ± 0,000 | 0,630 ± 0,000 | 0,541 | 0,754 |
| Random Forest | 0,848 ± 0,000 | 0,654 ± 0,003 | 0,630 ± 0,003 | 0,558 | 0,724 |
| **LightGBM (produção)** | **0,856 ± 0,000** | **0,674 ± 0,001** | **0,639 ± 0,001** | 0,550 | 0,763 |

Teste t pareado (F1 por seed): LightGBM vs. Regressão Logística **p < 0,001**;
vs. Random Forest **p = 0,013**. Threshold de operação tunado na validação:
0,261 (modelo calibrado, sem class weights — Brier 0,132 no teste).

**H1 (LightGBM supera baselines):** confirmada com significância, com a
ressalva honesta de que a margem é ~1 p.p. de AUC — o problema é quase
linear e a primeira configuração (árvores profundas default) *perdia* para a
Regressão Logística. A varredura de hiperparâmetros na validação foi
decisiva (ADR-0001).

## 2. Detecção de drift (cenário moderate, injeção no dia 15)

| Métrica | Valor | Meta |
|---|---|---|
| Dia de detecção (1ª severidade CRITICAL) | **15** (mesmo dia da injeção) | — |
| **Latência de detecção** | **0 dias** | **≤ 3 dias ✅** |
| Falsos alarmes CRITICAL pré-injeção (14 dias) | **0** | — |

No dia 15, PSI crítico em `MonthlyCharges`, `TotalCharges`, `tenure` e
`Contract` simultaneamente (relatório Evidently `evidently_*_day15_critical.html`).

## 3. Estático vs. retreinamento automático (30 dias simulados)

| Métrica | Pré-drift (D1–14) | Pós-drift estático | Pós-retreino automático (D20–30) |
|---|---|---|---|
| AUC-ROC | 0,792 | 0,754 | 0,756 |
| Brier (calibração; menor=melhor) | 0,145 | 0,207 | **0,201** (melhor que o estático em 10/11 dias) |
| F1 | 0,559 | 0,676 | **0,683** |
| Probabilidade média prevista | 0,24 (real 0,24) | 0,36 (real ≈ **0,45**) | **0,41** (real ≈ 0,45) |
| Alertas CRITICAL após adaptação | — | contínuos (D15–30) | **zero** (referência atualizada) |

Retreino único no dia 19 (aguardou 4 lotes rotulados do episódio — rótulos
chegam com 1 dia de atraso), registrado como versão 2 no MLflow Model
Registry com alias `@production`.

**Leitura dos resultados (importante):**

- **H2 confirmada nas métricas certas:** o F1 *sobe* pós-drift porque a taxa
  base quase dobra (24% → 44%) — métrica confundida pela prevalência. A
  degradação real aparece em AUC (−3,8 p.p.) e Brier (+0,062, +43%).
- **H4 refinada:** o retreino recupera **calibração** (proba média volta a
  acompanhar a taxa real; Brier melhor que o estático em 10 de 11 dias) e
  **silencia os alertas**, mas não recupera a AUC pré-drift — o regime novo
  (clientes homogeneizados em month-to-month/fibra) é intrinsecamente menos
  separável. Retreinar não recria sinal que deixou de existir; esse teto é
  visível ao comparar com o próprio estático no mesmo período.

## 4. Ablação: intensidade do drift × detecção (modo estático)

| Intensidade | Δ taxa de churn | Latência | Falsos alarmes | ΔAUC | ΔBrier |
|---|---|---|---|---|---|
| light (mensalidade +10%, odds ×1,2) | 24% → 31% | **0 d** | 0 | −0,019 | +0,023 |
| moderate (+25%, odds ×1,6) | 24% → 44% | **0 d** | 0 | −0,038 | +0,062 |
| severe (+45%, odds ×2,2) | 24% → 60% | **0 d** | 0 | −0,063 | +0,080 |

Mesmo o drift *light* é detectado no próprio dia, sem falsos alarmes em
nenhum cenário: com lotes de 250 clientes/dia, o PSI com limiares de Siddiqi
tem sensibilidade de sobra para a meta de 3 dias. A degradação escala
monotonicamente com a intensidade, como esperado.

## 5. Análise de erros (modelo de produção, teste)

Ver `artifacts/model/feature_importance.csv` e `reports/eda/`. Padrões:
falsos negativos concentram-se em clientes de contrato longo que cancelam
(evento raro e pouco sinalizado pelas features); `Contract`, `tenure` e
`MonthlyCharges` dominam a importância — exatamente as features alvo do
cenário de drift.

## 6. Limitações

1. Operadora fictícia (IBM sample data) — sem generalização direta.
2. Rótulos da simulação vêm de um oráculo calibrado; drifts reais podem ser
   mais abruptos/estruturados que os gerados por copula.
3. Detecção cobre P(X); estimar degradação sem rótulos (P(Y|X)) é trabalho
   futuro (NannyML).
4. Lotes diários, não streaming — migração para Kafka/Flink mantém o motor
   de decisão (ADR-0005).
