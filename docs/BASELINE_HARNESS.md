# Unified trainable-baseline harness

`baseline-harness-v1` independently trains `emer_like`, `cmr_like` and `paretojev`
using one data contract and one evaluator. The first two are explicit project
variants. They are **not reproductions of the EMER or CMR papers**, and a win here
does not establish superiority over those published systems.

## Run without external inputs

```bash
paretojev comparison-synthetic-data data/processed/comparison_synthetic \
  --seed 123 --train-slates 96 --validation-slates 48 --test-slates 48 \
  --slate-size 8 --objectives 3 --feature-dim 6

paretojev compare-baselines data/processed/comparison_synthetic/manifest.json \
  --output artifacts/comparison_synthetic \
  --config configs/comparison_smoke.yaml \
  --targets configs/targets_comparison.yaml
```

Use new, empty directories for both generation and a new experiment. The generator
is also available as
`paretojev.data.comparison.write_synthetic_comparison_data(directory, seed=123,
sizes=(96,48,48), slate_size=8, n_objectives=3, feature_dim=6)`.
Its final objective is minimized. It writes observed outcomes, padded candidates,
identities and strictly separated timestamp ranges; latent simulator probabilities
are excluded. These are protocol fixtures, not KuaiRand measurements.

The configuration has `training`, `evaluation`, `seeds`, `bootstrap_samples` and
`cpu_threads` sections/keys. Training controls common epochs, batch size, AdamW
settings, set-encoder dimensions, dropout and device; the registered `seeds` list
controls training randomness. Evaluation fixes top-K, uniform/one-hot/optional
pairwise/seeded Dirichlet preferences, cold quantile and Monte Carlo sample count.
Change these before registering a run, using validation evidence rather than test
results. A smoke budget is not a final research budget. The CLI returns `0` when
the experiment completes; use `check-targets` for a strict gate (exit `0` only when
passed, `3` for failed/not-evaluable required targets).

## What is compared

All methods receive the same candidate features, behavior/Jev probabilities,
confidence, entropy, popularity, history length and request preferences. Labels
are training/evaluation outcomes, never inference features. Negative objectives
become `1 - label` for utility losses while raw probability semantics remain intact.
All three served ranking scores receive pairwise logistic supervision from the
request's preference-weighted observed utility, including non-dominated trade-off
pairs. All objective heads also receive utility Brier calibration.

| Method | Architecture and additional training rule |
|---|---|
| `cmr_like` | Raw-signal set Transformer with preference FiLM; served score is the preference-weighted sum of objective logits. Per-objective pairwise losses are weighted by the request preference. |
| `emer_like` | Same raw-signal set encoder, plus a shared score head. A training-only evaluator divides detached current mean objective losses by their preceding EMA, clips multipliers to `[0.5,2]`, and renormalizes request preferences. First update uses unchanged preferences; EMA update rate is `0.1`. These weights supervise per-objective shared-score losses. EMA state is checkpointed and unused at inference. |
| `paretojev` | Existing confidence-aware fusion, preference FiLM, set Transformer, shared score and variance heads. Additional losses are observed Pareto-dominance ranking, per-request augmented Tchebycheff of shared-score objective losses (`rho=0.05`), Gaussian pairwise uncertainty, Jev utility-distribution KL, and raw-probability fusion calibration. |

The EMER-like feedback is a loss-based surrogate, not the published self-evolving
evaluator or RAS pipeline. Full ParetoJev in this harness does not add that EMA
controller. `COMPARISON_LOSS_WEIGHTS` in `models/comparison.py` fixes coefficients:
preference ranking `1`, objective calibration `0.25`, EMER feedback ranking `1`,
CMR objective ranking `1`, Pareto ranking `1`, Tchebycheff `1`, uncertainty `0.25`,
Jev distillation `0.35`, fusion calibration `0.5`. Only applicable terms are used.

## Fairness and selection

Each registered seed gives all methods the same training rows/order, preference
inputs, epoch count, batch size, optimizer settings and gradient clipping (`5`).
Set-encoder capacity settings are shared. Parameter counts and elapsed training
time are reported: equal optimizer budgets do **not** mean equal parameters,
FLOPs or wall time. No hyperparameter search is hidden in the runner.

Each method selects its checkpoint by maximum **validation** mean
preference-weighted observed utility over the fixed preference grid. Epoch zero
is eligible; exact ties retain the earliest checkpoint. Every configured epoch is
still executed. Validation must cover all objectives. The selected checkpoint is
saved/reloaded and its predictions checked for exact equality on a validation
probe. Test labels never choose checkpoints, preferences, comparator or cold cutoff.

Coldness uses the registered quantile of valid **training** candidate popularity;
that cutoff is frozen for validation/test interpretation. The two predeclared
comparators are `emer_like` and `cmr_like`: the primary `kuairand.paretojev`
namespace uses EMER-like, while `kuairand.paretojev_vs_cmr` uses CMR-like. Main gates
require hypervolume gain ≥2%, cold NDCG gain ≥3% and regret ≤0.005 against each,
plus ParetoJev controllability ≥0.60. The runner does not select a best-on-test
comparator. A broader claim against the strongest external system needs additional
protocols.

## Evaluation and uncertainty

All selected checkpoints see the same test rows, candidate order/masks and runtime
preference grid in one traversal. Ties retain candidate order. The shared utility
space is `[0,1]^M`:

- Positive objectives use conditional NDCG@K over slates with positive ideal DCG.
- Minimized objectives use one minus pooled selected-candidate risk; variable
  valid slate lengths retain their actual denominator.
- Cold NDCG uses the ordinary full ranking with relevance restricted to cold
  candidates. It averages covered positive objectives and reports each coverage.
- Hypervolume uses common Monte Carlo points in the unit cube and is unavailable
  if any objective lacks coverage.
- Relative gain is undefined when its baseline is zero. Undefined values are JSON
  `null`, not zeros or infinities.
- Worst-objective regret is the largest positive utility shortfall of ParetoJev's
  coordinate-wise best over the fixed grid against the declared comparator’s
  coordinate-wise best. It is neither oracle regret nor a per-preference loss.
- Controllability is Spearman correlation between each requested weight and its
  achieved utility over the grid; constant utility has zero correlation by convention.

Paired user-cluster bootstrap intervals cover objective mean and cold-NDCG absolute
differences for **fixed fitted checkpoints and preferences**. Fewer than two user
groups make intervals unavailable; an undefined replicate withholds its interval
and reports the valid count. No hypervolume data-bootstrap interval is produced.
The hypervolume standard error describes integration error only, not uncertainty
from users or training. Near-threshold gains need more integration precision and
appropriate paired uncertainty analysis before a research claim.

The summary averages each metric over **all registered seeds**; if any seed is
undefined, that aggregate is null. Seed means are descriptive point estimates,
not a significance test. Passing numerical targets alone is not a paper claim.

## Registration, outputs and resume

`registry.json` is written before training. Registration includes the manifest and
split hashes, protocol, methods, exact budgets/seeds, objective ordering, selection
rule, training-fitted cold cutoff, target configuration, Python/package/platform
identity and hashes of package source files. `run_id` hashes that registration.
Failures remain recorded. Each completed checkpoint and training report is hashed.

Each `seed-<seed>/` contains three checkpoints and training histories, an
`evaluation.json`, `predictions.npz` with method/preference/slate audit arrays, and
`prediction_identity.npz` with source identities. `summary.json` contains seed
means, method parameter counts, evidence, engineering checks and target status.

```bash
paretojev compare-baselines data/processed/comparison_synthetic/manifest.json \
  --output artifacts/comparison_synthetic \
  --config configs/comparison_smoke.yaml \
  --targets configs/targets_comparison.yaml --resume
```

Resume validates registration and completed artifact hashes, reloads completed
methods, and trains missing methods. It is a method-level restart, not mid-epoch
optimizer recovery. Changed source, inputs, environment, targets or settings require
a new output directory. Test reports are recomputed from registered checkpoints.

An operating-system advisory lock rejects simultaneous writers to one output
directory and releases after a process exits or crashes. Its hidden sidecar file
is intentionally retained. Package source hashes are checked again before
registering completed training and evaluation results; edit source between runs,
not during a registered run. Deterministic PyTorch algorithms are required, so an
unsupported nondeterministic kernel fails instead of silently weakening replay.

`requirements-lock.txt` records the Python dependencies used on macOS arm64 with
Python 3.12 for this validation. Install that snapshot before `pip install -e .
--no-deps` for the same environment; other hardware/platforms need their own
registered environment and are not promised identical numerical outputs.

## Real-data handoff: minimum inputs

A real run needs three complete NPZ files, one manifest, and four upstream evidence
documents. Prepared slates alone are insufficient: behavior and decision scores
must already be aligned. See [NPZ_CONTRACT.md](NPZ_CONTRACT.md) for tensor metadata.

```json
{
  "protocol_version": "baseline-harness-v1",
  "dataset": "kuairand",
  "data_kind": "real",
  "splits": {
    "train": {"path": "train.npz", "sha256": "<file SHA-256>"},
    "validation": {"path": "validation.npz", "sha256": "<file SHA-256>"},
    "test": {"path": "test.npz", "sha256": "<file SHA-256>"}
  },
  "upstream": {
    "feature_spec": {"path": "feature_spec.json", "sha256": "<file SHA-256>"},
    "behavior_provenance": {"path": "behavior_provenance.json", "sha256": "<file SHA-256>"},
    "decision_provenance": {"path": "decision_provenance.json", "sha256": "<file SHA-256>"},
    "evaluation_protocol": {"path": "evaluation_protocol.json", "sha256": "<file SHA-256>"}
  }
}
```

Paths resolve relative to the manifest. All file hashes must match. The canonical
JSON hash of `feature_spec` must match all NPZ `feature_spec_hash` values. Split
valid timestamps must satisfy `max(train) < min(validation)` and
`max(validation) < min(test)`, with common dimensions and objective/feature order.
Real comparison currently accepts `dataset=kuairand` only.

The upstream documents should identify train-fitted feature transformations;
behavior training/selection/scoring periods and checkpoint; decision model/version,
requests and score provenance; and the exposure track, slate construction, labels,
fixed evaluation population and sampling rules. Their hashes establish identity
and auditability; the loader does **not** independently establish authenticity or
validate every upstream leakage claim. Those require review of the supplied evidence.

If features are not prepared, the minimum source handoff is KuaiRand standard and
random logs plus caption/category metadata and either aligned versioned decision
scores or a usable decision-model endpoint. The behavior checkpoint can be trained
locally from the supplied logs; an existing checkpoint or aligned behavior scores
can shorten that work. Already scored artifacts need no live API. Raw datasets and credentials stay out
of Git. Without these inputs, continue engineering and synthetic validation and
record real gates as `not_evaluable`.
