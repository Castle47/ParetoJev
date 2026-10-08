# Project Status

**Updated:** 2026-10-08

## Completed

- Phase 0 requirements, architecture and target gates
- installable Python package and CLI
- MicroLens official/legacy schema aliases and timestamp-unit normalization
- MicroLens upstream-compatible occurrence-weighted candidate sampler
- explicit tie policies, including upstream deterministic jitter and stable research mode
- durable/resumable TypeSafe/open-JEV scoring with signature validation, cache and run reports
- Choice, pointwise Noul and slate-conditioned Noul protocols
- MicroLens baselines, reciprocal-rank fusion and cross-validated simplex hybrid
- nested user-CV logistic stacker with raw/rank/margin and confidence interactions
- out-of-fold Choice temperature scaling and Noul Platt calibration
- user-bootstrap intervals, paired reference tests and model-complementarity diagnostics
- Phase A target file and one-command local execution scripts
- KuaiRand pseudo-slate/content manifest pipeline
- leakage-safe prior item/user counts for cold-start and history buckets
- balanced KuaiRand Jev sampler across popularity, history, exposure and feedback-density cells
- strict pre-slate historical-Bayes bootstrap with overlap de-duplication and prefix-period initialization
- behavior-score interchange contract plus strict Jev/behavior/candidate/objective alignment
- train-fitted KuaiRand candidate feature specification and feature-complete NPZ materialization
- chronological equal-timestamp-safe KuaiRand split utility with cumulative prefix files
- formal SASRec + MMoE behavior training, early stopping, OOV-aware checkpoint and strict slate scorer
- behavior-informed Jev annotation ordering using entropy, margin uncertainty and cross-objective winner disagreement
- confidence-aware fusion and preference-conditioned set reranker
- Pareto/Tchebycheff/uncertainty/distillation losses
- checkpoint-level runtime-preference sweep with shared-box Pareto hypervolume, controllability, cold-item and regret diagnostics
- precomputed real-data training contract
- **49 passing tests** plus successful synthetic MicroLens, KuaiRand, materialization and checkpoint-evaluation CLI integration runs
- passing synthetic model smoke and target report

## Active next phase

**Phase A — MicroLens reproduction and hybrid win**

The code path is now ready for a real, reproducible run. Required external inputs for
real benchmark numbers are:

1. official MicroLens pairs/title files;
2. official Jev API key or a reachable self-hosted JEV endpoint;
3. optional GPU endpoint for the JEV-27B-VL cover track.

The current execution environment cannot fetch the binary dataset and has no Jev
credential or GPU endpoint. Therefore no real MicroLens, KuaiRand or Jev benchmark
number is recorded in this repository. Synthetic integration output is used only to
validate code and report structure.

## Go/no-go sequence

1. Reproduce item-CF, popularity and title-similarity baselines under the exact public protocol.
2. Score official Jev Choice(click) and pointwise Noul(click), with full cost/latency provenance.
3. Verify probability calibration and measure CF/Jev error complementarity.
4. Run RRF, outer-CV simplex fusion and nested-CV stacking.
5. Require both absolute Phase A targets and paired-bootstrap evidence versus reproduced item-CF.
6. Run the implemented strict temporal split, formal SASRec+MMoE trainer/scorer, stratified KuaiRand subset scorer and materializer.
7. Compare the formal neural branch with the historical-Bayes bootstrap and add PLE only if the first baseline is stable.
8. Only after teacher uplift survives robust evaluation, train the low-latency student.
