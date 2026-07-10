# ADR-0001 — LightGBM como modelo de produção

**Status:** aceito · **Data:** 2026-07-10

## Contexto

O pipeline precisa de um classificador binário de churn servível em CPU, com
baixa latência, treinável em segundos (o retreinamento automático roda dentro
do loop de produção) e com bom desempenho em dados tabulares mistos
(3 numéricas + 16 categóricas, ~5k linhas de treino).

## Decisão

LightGBM (`LGBMClassifier`), com suporte nativo a features categóricas
(sem one-hot), árvores rasas (`num_leaves=7`, `max_depth=3`) e regularização
forte — hiperparâmetros escolhidos por varredura na validação. **Sem**
`class_weight="balanced"`: a calibração fica muito melhor (Brier 0,132 vs
0,159, mesma AUC/F1 no teste) e o desbalanceamento é tratado pelo threshold
de decisão, tunado na validação e versionado junto ao modelo.

## Alternativas consideradas

- **Regressão Logística** — surpreendentemente competitiva (AUC 0,849 vs
  0,856): o problema é quase linear. Mantida como baseline; perdeu com
  significância estatística (teste t pareado, p < 0,05) após o tuning.
- **XGBoost / CatBoost** — desempenho esperado equivalente; LightGBM tem
  treino mais rápido nesta escala e categóricas nativas sem preparo extra.
- **Redes neurais (TabNet etc.)** — inadequadas para ~5k linhas; custo de
  treino incompatível com retreino automático frequente.

## Consequências

- Treino completo em < 5 s ⇒ retreinamento automático barato.
- Modelo com melhor desempenho **e** boa calibração, essencial para que o
  time de retenção priorize clientes por probabilidade real de churn.
- A configuração inicial (árvores profundas default) perdia para a Regressão
  Logística — registrado como lição: em datasets pequenos, baseline linear
  forte é obrigatório para calibrar expectativas.
