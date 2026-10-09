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
- independently trained EMER-like, CMR-like and full ParetoJev harness variants on one strict data protocol
- preregistered seeds/budgets, validation-only selection and both fixed baseline comparisons
- shared evaluation, per-slate prediction audit, user-cluster bootstrap and undefined-metric handling
- method-level resume with artifact/source hashes, concurrent-writer lock and deterministic training
- real-data evidence requirements which synthetic results cannot satisfy
- **209 passing tests** on the local Python 3.12 environment
- passing synthetic model smoke and target report
- real official MicroLens acquisition and 200-user item-CF/title baseline reproduction
- 20-user / 400-candidate real AutoTrust JEV-27B pilot, revision-checked and cached
- extension to all original 200 users / 4,000 real decisions, with exact pilot reuse
- full primary and fresh-180-user evaluation, raw-response audit and tie-sensitivity diagnosis
- new 200-user evaluation disjoint from the earlier cohort; CF refitted excluding all 400 users
- frozen Jev tie-breaking versus CF and TF-IDF control passes both predeclared AUC tests
- zero-HTTP scoring replay and identical raw-input regeneration
- real SASRec-style next-item training: 57,845 users, separate 1,000-user checkpoint validation
- frozen MiniLM semantic control, cached 400-user developmental fusion comparison
- raw split/history audit and exact offline checkpoint/evaluation replay
- fixed 700-user confirmation completed: 14,000 genuine scores, 1/2 AUC gates passed
- Jev improves SASRec AUC but superiority over MiniLM fusion remains unconfirmed

## Current research boundary

The baseline harness engineering phase is complete. The final registered synthetic
comparison and replay evidence are in the [experiment report](../reports/synthetic-baseline-2026-10-08/README.md)
and execution log. These exercise
training, evaluation and provenance; they do not establish a ParetoJev research win.
The three-seed synthetic run showed weaker ParetoJev hypervolume and
controllability despite improved cold-NDCG, so no superiority claim is made and no
configuration is selected or changed based on those test results.

The [full real MicroLens run](../reports/microlens-real-200-2026-10-08/README.md)
supports an AUC increment for CV small-weight fusion; its paired interval is
positive under both primary jitter and post-hoc conventional half-credit AUC.
Main targets pass 2/4: AUC 0.7274 and HR@5 0.570 remain below thresholds. HR/NDCG
improvement intervals still cross zero. Gains concentrate in behavior-score ties.
The [new-user validation](../reports/microlens-independent-2026-10-08/README.md)
has now tested that hypothesis without new-user tuning: Jev tie AUC 0.7176 versus
CF 0.6650 and title tie AUC 0.6738, both predeclared 97.5% gain intervals positive.
The call plan saves only 5.45%, and the 5% fusion and tie rule have identical weak
orders on these users. That finding motivated stronger behavior and cheap semantic controls, without
crediting untested neural ParetoJev complexity. The [SASRec/MiniLM development experiment](../reports/microlens-sasrec-development-2026-10-08/README.md)
now supplies both controls. SASRec validation NDCG@10 is 0.5480 versus CF 0.4854.
On the already examined 400 users, AUC is 0.7622 for SASRec and 0.7701 for fixed
20% Jev fusion; the matched MiniLM fusion is 0.7578. The 5% rules barely change
ordering and fail the two increment gates; only the behavior-strength gate passes.
The 20% diagnostic is promising, but NDCG gain versus SASRec remains uncertain,
and no independent confirmation or multi-objective ParetoJev claim follows.
The original full project gates are not all achieved. The legacy
smoke targets remain engineering probes with privileged
simulator probabilities; the comparison harness deliberately forbids those targets.

## External inputs for real experiments

**Phase B — fixed KuaiRand baseline comparison**

Smallest ready-to-run handoff: chronological train/validation/test feature NPZs,
their manifest and the four provenance documents described in
[BASELINE_HARNESS.md](BASELINE_HARNESS.md#real-data-handoff-minimum-inputs).
This route needs no live API or new behavior checkpoint. Alternatively provide the
raw KuaiRand logs/metadata plus a reachable Jev endpoint or aligned cached decision
scores; the behavior model can be trained locally. Credentials belong in environment
variables, never in Git or chat. Formal random-exposure-track consistency and
statistical evidence remain prerequisites for a broad research claim.

**Phase A — MicroLens reproduction and hybrid win**

Official pairs/title files have been downloaded and hashed. The first real pilot
uses AutoTrust's public demo endpoint with pinned deployment revision, one concurrent
request and a 400-decision cap. No user API key or paid compute was needed. The
local 8GB M1 cannot run the planned unquantized 27B deployment; no weights were downloaded.

The 200-user Jev evaluation is complete: 400 pilot decisions reused, 3,600 new calls,
0 failed slates, approximately 60.3 minutes at the registered one-request-per-second
start limit. The public demo is not a guaranteed bulk service. A larger controlled
experiment can use a suitable endpoint, cached real scores, or dedicated model
deployment. Native cover-image scoring needs the actual VL pipeline. Serving
targets need a declared target device; no GPU latency benchmark is claimed.

## Go/no-go sequence

1. Reproduce item-CF, popularity and title-similarity baselines under the exact public protocol.
2. Score official Jev Choice(click) and pointwise Noul(click), with full cost/latency provenance.
3. Verify probability calibration and measure CF/Jev error complementarity.
4. Run RRF, outer-CV simplex fusion and nested-CV stacking.
5. Require both absolute Phase A targets and paired-bootstrap evidence versus reproduced item-CF.
6. Run the implemented strict temporal split, formal SASRec+MMoE trainer/scorer, stratified KuaiRand subset scorer and materializer.
7. Compare the formal neural branch with the historical-Bayes bootstrap and add PLE only if the first baseline is stable.
8. Only after teacher uplift survives robust evaluation, train the low-latency student.

## Latest confirmation and next decision

The [fixed 700-user study](../reports/microlens-sasrec-confirmatory-2026-10-09/README.md)
completed with no outcome-driven changes. AUC: SASRec 0.779549; MiniLM fusion
0.783083; Jev fusion 0.785301. Jev minus SASRec is +0.005752 with predeclared
97.5% interval [0.000413, 0.011128]. Jev minus MiniLM is +0.002218 with interval
[-0.003459, 0.007895]. The joint confirmation gate fails (1/2), and HR/NDCG gains
are uncertain. Do not expand this cohort or change weights to turn that result
into a pass. One TLS EOF was handled by a separate, documented operational
amendment allowing up to 20 extra transport attempts; exactly one was needed.

Prioritize the local semantic baseline, comparable deployment-cost measurements
and real multi-objective data requirements before increasing model complexity
or Jev collection. No unresolved input blocks the completed study. The original
KuaiRand/ParetoJev and serving research gates remain unevaluated.
