# 2026 World Cup — Forecast Lock

**Generated:** 2026-06-11 03:34:07 UTC
**Model trained on international results through:** 2026-06-09
**Simulations:** 10,000 (seed 42)

These predictions were committed **before the tournament's first match (11 June 2026)**.
The git commit timestamp is the verifiable record that they predate every result.

## Method (one paragraph)
A rolling Elo rating built from scratch over every international since 1872 serves as the
benchmark. A Poisson goals model with team attack/defence strengths, a neutral-venue flag,
exponential time-decay weighting and a Dixon-Coles low-score correction predicts expected
goals per side; a Monte Carlo simulator plays the full 48-team bracket 10,000 times.
Walk-forward backtesting (2023–2026) showed the goals model beats the Elo baseline on RPS
and log-loss (Wilcoxon p<1e-4).

## Title odds — top 5 at lock time
  1. Argentina — 20.5%
  2. Spain — 15.6%
  3. Brazil — 9.9%
  4. England — 9.0%
  5. France — 6.5%

## Honesty caveats
- The exact Round-of-32 slot mapping (FIFA's third-place lookup table) is approximated by a
  reproducible structural pairing; it does not materially affect aggregate title odds.
- Knockout ties are resolved by the model's conditional win probability (ET/penalties proxy).
- The model rates teams from results only; it cannot see injuries, squad rotation, or form news.
