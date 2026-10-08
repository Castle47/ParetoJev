# Execution Log

## 2026-10-08 — Phase 0 bootstrap

### Environment observed

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- pytest 9.0.2

### Completed

1. Created Git repository and Python package.
2. Implemented data, model, Jev, loss, metric and target modules.
3. Added CLI, configs, docs and CI.
4. Ran compile check.
5. Ran `pytest`: **14 passed**.
6. Ran end-to-end synthetic smoke experiment.

### Smoke result

- initial held-out loss: `1.06637`
- final held-out loss: `0.96978`
- reduction: `9.06%`
- behavior Brier: `0.03795`
- Jev-like Brier: `0.01665`
- fused Brier: `0.01156`
- cold mean Jev gate: `0.65949`
- warm mean Jev gate: `0.65611`
- status: passed

The cold/warm gate separation is small but in the intended direction. This is sufficient for a smoke test, not for a model claim.

### Not executed

- MicroLens download and real baseline reproduction;
- official Jev API calls;
- self-hosted JEV-27B/VL inference;
- KuaiRand preparation and training;
- GPU latency benchmark.

Reason: the execution container did not have the datasets, API credentials, a running open-JEV endpoint or GPU. No real metrics were fabricated.

### Additional verification

- Editable installation succeeded with local build isolation disabled after switching the build backend to installed `setuptools`.
- CLI help and all registered commands loaded successfully.
- Final test count after adding the precomputed-data training path: **14 passed**.
- A direct attempt to download the official MicroLens pairs file from this container failed because outbound file download was unavailable; the repository therefore records no fabricated real-data result.

## 2026-10-08 — Phase A reproducibility hardening

### Protocol audit and corrections

- Matched the public MicroLens demo's occurrence-weighted negative sampler instead of
  silently sampling uniformly from deduplicated items.
- Matched the public demo's right-open time-window boundary.
- Added an explicit `upstream_jitter` tie policy and retained `stable` and
  independently randomized alternatives.
- Made pointwise Noul genuinely pointwise by hiding competing candidates from state;
  candidate content is carried in the individual question.

### Scoring reliability

- Added durable JSONL append, run signatures, resume, overwrite protection, error logs,
  latency/usage aggregation and provenance fingerprints.
- Resuming an output now rejects provider/model/mode/objective/scope mismatches.
- Added objective sub-selection so MicroLens Phase A can request only `click` instead
  of paying for unrelated multi-objective questions.

### Research-grade Phase A evaluator

- Added reciprocal-rank fusion as a label-free baseline.
- Added nested user-CV logistic stacking with raw score, within-slate rank, margin,
  and pair interactions. Raw interactions allow slate-level Jev confidence/entropy
  to modulate candidate-level Jev evidence.
- Added out-of-fold Choice temperature scaling and Noul Platt calibration.
- Added Brier/ECE/NLL diagnostics and reliability curves.
- Added user-bootstrap confidence intervals, paired differences versus a declared
  reference and model-complementarity/oracle diagnostics.
- Added `configs/targets_microlens.yaml` and a complete Phase A shell workflow.

### Verification

- `python -m compileall -q src tests`: passed.
- `pytest -q`: **24 passed**.
- Synthetic 30-user/20-candidate CLI run completed through calibration, RRF, outer-CV
  hybrid, nested-CV stacker, bootstrap, pairwise comparison, complementarity and
  provenance output.
- Synthetic values are deliberately not committed as research outcomes.
