# ADR-0003 — Copula gaussiana interna como engine default de dados sintéticos (SDV opcional)

**Status:** aceito · **Data:** 2026-07-10

## Contexto

A especificação original sugere SDV (Synthetic Data Vault) para gerar o
drift sintético. O SDV, porém, depende de `ctgan`/PyTorch — ~2 GB de
dependências — para um projeto que usa apenas o `GaussianCopulaSynthesizer`
(estatístico, sem redes neurais). Isso pesa na CI, no Docker e na
instalação de quem for corrigir o projeto.

## Decisão

Implementar internamente uma **copula gaussiana para dados mistos**
(`churn_mlops/drift/synthesizer.py`, ~80 linhas): marginais empíricas para
numéricas, frequências para categóricas, acopladas pela matriz de correlação
no espaço normal — o mesmo princípio matemático do sintetizador homônimo do
SDV (Patki et al., 2016). O SDV permanece disponível como engine alternativo
(`uv sync --extra sdv` + `synthesizer: sdv` no config), preservando a
fidelidade à especificação.

Complemento necessário em ambos os engines: `enforce_structural_consistency`
impõe as regras rígidas do domínio que copulas não capturam (ex.:
`InternetService == "No"` ⇔ add-ons `== "No internet service"`) — bug real
encontrado pelos testes unitários.

## Alternativas consideradas

- **SDV como dependência obrigatória** — fidelidade total à especificação ao
  custo de ~2 GB; rejeitado como default, mantido como extra.
- **Bootstrap com perturbação (sem copula)** — simples, mas não preserva as
  correlações entre features (ex.: tenure × TotalCharges, r ≈ 0,89 mantido
  pela copula — verificado em teste).

## Consequências

- `uv sync` completo em ~1 min; imagem Docker e CI enxutas.
- Testes unitários cobrem o sintetizador (schema, determinismo por seed,
  preservação de correlação, consistência estrutural).
- Trade-off documentado: a copula interna não modela distribuições
  multimodais tão bem quanto CTGAN — irrelevante aqui, pois o objetivo é
  drift *controlado*, não fidelidade máxima.
