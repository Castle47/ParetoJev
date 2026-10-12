# Aligned behavior comparison — registered before fitting

Date: 2026-10-12. This supersedes the planned rare-feedback optimization as the
next experiment. No new sampling, rare-positive enrichment, label changes or gate
relaxation. Prior v1/v2 results remain frozen.

## Shared protocol

Reuse the exact v2 target rows, order, seven labels and inverse inclusion weights:
1,699,506 train targets before April 18; 188,642 calibration targets on April 18;
187,419 evaluation targets on April 19. No new sample seed or fresh evaluation
claim. April 19 was previously inspected and remains development data.

Reuse the frozen v2 upload-metadata tree (64 iterations, seven independent heads)
without refitting: its training data, calibration algorithm, and evaluation are
already the intended protocol. Verify manifests, model and prediction hashes,
and recompute predictions before comparing.

Fit two neural controls on the same 26 derived fields:

- Tabular MMoE: identical available source information to the tree. Train-only
  weighted mean/std normalization; zero-imputed normalized values plus missing
  indicators. Missingness is already visible to the tree. Tab uses an embedding.
- SASRec-style + MMoE: same tabular branch plus candidate and last-50 exposure item
  embeddings, causal sequence encoder, and train-known user embeddings. This
  adds ordered IDs, so this is a richer information package, NOT a pure architecture
  comparison against the tree. The tabular arm prevents conflating this difference.

The neural arms use four experts, seven gates/heads, seed 20261012, AdamW 0.001,
three fixed epochs, batch 512, gradient clipping 5, no class reweighting, and loss
weighted by inverse inclusion probability normalized to mean one. Final epoch
only; no checkpoint selection on either later period. Same positive-slope Platt
routine/config as v2 on April 18 only. Freeze both models and calibrators before
loading April 19 labels in the training/evaluation runner. Same CPU deterministic
execution and 1,000 user-cluster bootstrap replicates.

Keep the original SASRec-style encoder (32 dimensions, 1 layer, 2 heads) and fixed
65,536 item-hash buckets. History contains exposures, not positive-only histories;
all same-time events excluded. Reconstruct targets using the original stable sort,
seed/user/period sampler; verify every user/time/label/weight against frozen v2.
Store raw event record indices so timestamp ties cannot confuse row identity.

## Evaluation and interpretation

Report raw/calibrated AUC, log loss, Brier, ECE, probability means and per-scene
metrics using the shared evaluator. Retain the existing seven gates: >=20 sampled
positives, AUC>.5, calibrated log loss AND Brier below training tab prevalence.
Report all failures. Pairwise calibrated log-loss user bootstrap: tabular MMoE
minus tree, SASRec MMoE minus tree, SASRec MMoE minus tabular MMoE; exploratory,
conditional on these fitted models, no multiplicity correction. Do not infer
convergence or general architecture superiority from a single fixed-budget seed.
Neither three neural epochs vs 64 tree iterations nor wall times are equal compute
budgets; both are disclosed. No tuning to make a desired model win.

Upload metadata still assumes static-at-upload fields without timestamp proof.
No user profile snapshots, monthly aggregates, outcome features or Jev are used.
No current ParetoJev / real Jev increment is established. Research target mode
stays false. Rare-feedback work is deferred until after this comparison.
