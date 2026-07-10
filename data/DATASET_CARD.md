# Data Card — Telco Customer Churn

| Atributo | Valor |
|---|---|
| **Nome** | Telco Customer Churn |
| **Fonte primária** | [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), origem nos IBM Sample Data Sets |
| **Mirror usado pelo download** | [treselle-systems/customer_churn_analysis](https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv) (cópia byte-a-byte do CSV do Kaggle; evita autenticação da API do Kaggle) |
| **Licença** | CC0 — Domínio Público |
| **Volume** | 7.043 registros × 21 colunas |
| **SHA-256** | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` (verificado a cada download por `churn_mlops/data/download.py`; download falha se divergir) |
| **Variável-alvo** | `Churn` (Yes/No) — taxa de 26,5% (desbalanceamento moderado) |

## Schema

- **Identificação**: `customerID` (descartado no pré-processamento)
- **Perfil**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Conta**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Serviços**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

## Qualidade e tratamento

- `TotalCharges` chega como string com 11 valores em branco — todos de
  clientes com `tenure == 0`; convertidos para `0.0` (cliente novo, nada
  faturado ainda). Tratamento reproduzível em `churn_mlops/data/preprocess.py`.
- `SeniorCitizen` chega como 0/1; normalizado para `No`/`Yes` para tratamento
  uniforme como categórico (inclusive nos testes de drift qui-quadrado).
- Regras estruturais do domínio: `InternetService == "No"` ⇔ add-ons
  `== "No internet service"`; `PhoneService == "No"` ⇔
  `MultipleLines == "No phone service"`.

## Vieses e limitações conhecidas

- Operadora **fictícia** norte-americana (dados de demonstração da IBM): não
  generalizar para outras operadoras/mercados sem validação.
- Sem atributos geográficos ou socioeconômicos ⇒ análise de fairness por
  subgrupo limitada a gênero/senioridade.
- Desbalanceamento moderado (73,5% / 26,5%) ⇒ métricas sensíveis à classe
  minoritária (F1, PR-AUC) em vez de acurácia.
- Corte transversal (sem carimbo de tempo real): a dimensão temporal da
  simulação é sintética por construção.
