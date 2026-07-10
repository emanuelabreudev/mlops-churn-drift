# ADR-0004 — Evidently para relatórios; motor de decisão de drift próprio

**Status:** aceito · **Data:** 2026-07-10

## Contexto

O monitoramento tem dois consumidores distintos: (1) o **gatilho de
retreinamento**, que precisa de decisões determinísticas, testáveis e com
limiares explícitos; (2) **humanos**, que precisam de relatórios visuais
ricos por lote.

O Evidently AI mudou de API de forma incompatível duas vezes entre 2023 e
2025 (0.4 → 0.7). Acoplar o gatilho de produção à API de uma biblioteca em
mutação é risco operacional.

## Decisão

Separar decisão de visualização:

- **Decisão** (`monitoring/metrics.py`): implementação própria de PSI
  (bins por quantis da referência para numéricas; frequências para
  categóricas), KS (numéricas) e qui-quadrado (categóricas), com limiares da
  convenção de credit scoring (Siddiqi, 2017): PSI ≥ 0,25 crítico;
  0,10–0,25 moderado. Dataset drift ⇔ alguma feature crítica **ou** ≥ 50%
  das features moderadas. 100% coberto por testes unitários.
- **Visualização** (`monitoring/evidently_report.py`): Evidently
  `DataDriftPreset` gera os HTML interativos por lote, com import adiado e
  falha não-fatal — se a API mudar de novo, o pipeline perde o artefato
  visual, nunca a operação.

## Alternativas consideradas

- **Evidently também para a decisão** — menos código, mas o gatilho ficaria
  refém dos defaults/API da biblioteca e seria mais difícil de testar com
  limiares exatos.
- **Alibi Detect / NannyML** — bons detectores; NannyML brilha em estimação
  de performance *sem* rótulos (candidato a trabalho futuro), mas nenhum
  cobre relatórios visuais tão completos quanto o Evidently.

## Consequências

- O gatilho tem semântica exata documentada e testada (7 testes).
- Duplicação parcial de cálculo (PSI nosso + PSI do Evidently) — custo
  aceitável (< 1 s/lote) pelo desacoplamento.
