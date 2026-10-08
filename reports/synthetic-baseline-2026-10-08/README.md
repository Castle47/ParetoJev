# Registered synthetic baseline comparison — 2026-10-08

**Engineering validation only. No real recommendation benchmark or serving claim.**

Code was frozen at `5202bc35f1641a01b7aa4221fbcc67950bdfede2`, with a clean Git
checkout at registration. The bundle restored base was `26ff82c`. Runtime:
Python 3.12, local CPU, one PyTorch thread; full versions are in `registry.json`
and `../../requirements-lock.txt`. Source SHA-256, data hashes, fixed loss
weights and complete configuration are registered before training.

Data: 192 training / 96 validation / 96 held-out synthetic slates, width 12,
six objectives (five maximize, one minimize), ten features, synthetic generator
seed 123. Latent simulator probabilities are excluded from the training protocol.
All methods use seeds 7/17/29 and four training epochs. Positive metrics are
conditional NDCG@5; all methods share 30 preference vectors and 100,000 common
hypervolume integration points. The cold cutoff is fitted on training data.

| Method | Parameters | Mean synthetic hypervolume |
|---|---:|---:|
| EMER-like project variant | 26,215 | 0.036373 |
| CMR-like project variant | 26,166 | 0.045207 |
| Full ParetoJev harness variant | 26,529 | 0.034407 |

Across the fixed seeds, ParetoJev's mean relative hypervolume gain is **−5.39%**
versus EMER-like and **−23.75%** versus CMR-like. Mean cold-NDCG relative gains
are +6.17% and +10.40%, respectively; worst-objective regrets are 0.02940 and
0.05280. Its mean controllability is −0.14948. These are a small-budget synthetic
diagnostic and do not establish model superiority. No configuration was selected
or altered to improve the observed test results. The preliminary integration run
used the same registered data/budgets and produced the same metric values; the
final run additionally froze source and verified replay.

All seven real research gates correctly report **not_evaluable**, because evidence
is explicitly synthetic. `compare-baselines` exits zero on successful engineering
completion; `check-targets` returns 3 here. A point-estimate gate would still not
establish statistical significance: the reported hypervolume standard error is
integration error only. User-cluster bootstrap intervals in each seed report cover
fixed-checkpoint objective/cold differences, not training uncertainty or hypervolume.

## Verification

- 179 tests passed, including exact same-seed model/history reproduction and
  checkpoint invariance when only held-out labels change.
- Nine independent model/seed checkpoints trained and reloaded exactly.
- Resume reused every saved checkpoint; all nine checkpoint files, all three
  evaluation JSON files and the summary remained byte-for-byte identical.
- Regenerating all synthetic input splits reproduced identical SHA-256 hashes.
- Source compilation, shell syntax and Git whitespace checks passed.

`verification.json` and `tests.txt` record these checks. `registry.json` records
configuration/provenance. `seed-*.json` retain per-method metric points, training
histories, coverage and bootstrap reports. `summary.json` aggregates across seeds;
undefined values stay null. `research-targets.json` is the explicit gate result.

## Reproduce

From the repository root, install the recorded environment and run:

```bash
paretojev comparison-synthetic-data artifacts/comparison-data \
  --seed 123 --train-slates 192 --validation-slates 96 --test-slates 96 \
  --slate-size 12 --objectives 6 --feature-dim 10
paretojev compare-baselines artifacts/comparison-data/manifest.json \
  --output artifacts/comparison-verified \
  --config configs/comparison_smoke.yaml --targets configs/targets_comparison.yaml
paretojev compare-baselines artifacts/comparison-data/manifest.json \
  --output artifacts/comparison-verified \
  --config configs/comparison_smoke.yaml --targets configs/targets_comparison.yaml --resume
paretojev check-targets artifacts/comparison-verified/summary.json configs/targets_comparison.yaml
```

Use new directories for a new experiment; the generator does not overwrite files.
The last command intentionally exits 3 on synthetic evidence. Checkpoints and
prediction NPZs remain under `artifacts/comparison-verified/` in the local working
copy and are excluded from Git; the source archive can recreate them. Moving a run
or changing its input paths/environment requires a new registration.

Next research iteration must inspect validation-only preference response and loss
balance and register a fresh held-out comparison before tuning. Real research
requires the input contract in `docs/BASELINE_HARNESS.md`; synthetic results cannot
replace those inputs. Student/serving gates remain deferred until a real teacher
gain is established.
