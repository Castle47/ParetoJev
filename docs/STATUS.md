# Project Status

**Updated:** 2026-10-08

## Completed

- Phase 0 requirements, architecture and target gates
- Installable Python package and CLI
- MicroLens deterministic protocol + hybrid evaluation
- KuaiRand pseudo-slate/content manifest pipeline
- Official TypeSafe and open JEV clients with SQLite caching
- SASRec + MMoE behavior branch
- confidence-aware fusion and preference-conditioned set reranker
- Pareto/Tchebycheff/uncertainty/distillation losses
- precomputed real-data training contract
- 14 passing tests
- passing synthetic smoke and target report

## Active next phase

**Phase A — MicroLens reproduction and hybrid win**

Required external inputs:

1. official MicroLens pairs/title files;
2. official Jev API key or a reachable self-hosted JEV endpoint;
3. optional GPU endpoint for the JEV-27B-VL cover track.

The current execution environment had no outbound dataset download path, so Phase A real metrics were not run.

## Go/no-go sequence

1. Reproduce item-CF/title metrics.
2. Score official Jev Choice and Noul.
3. Measure complementarity and cross-validated hybrid.
4. Only if there is real uplift, expand to KuaiRand sampling.
5. Only if teacher uplift survives robust evaluation, train the student.
