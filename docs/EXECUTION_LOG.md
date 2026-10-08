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
