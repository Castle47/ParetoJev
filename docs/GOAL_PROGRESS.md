# Active user goal

Deliver verified overall multi-objective EMER+Jev superiority over EMER within the user's65 CNY total cap. Additional budget was explicitly declined; do not ask again or start new paidGPU inference. Goal is active and not achieved.

## Completed retrospective user-holdout validation

All72 fresh, equal-budget fits completed successfully. Main training session1364 and finalizer22319 both exited0; do not restart them. All36K existing real-score slates (12,606 users) received exactly one outer-fold prediction per arm/seed. All data and three fold-selection audits passed. Independent prediction and metric replay was exact.

Report: reports/emer-user-cv-v1/README.md. Protocol: docs/EMER_USER_CV.md.

- Mean six-positive NDCG@3:0.299540889 →0.303180590, +1.22%.
- Absolute aggregate difference+0.003639701; conditional paired-user95% interval[+0.000818641,+0.006560752].
- Each of three fixed seeds and each of three user folds has a positive aggregate point gain.
- Follow falls by0.005737264 absolute (−2.48% relative), exceeding the fixed0.005 maximum point drop. Other guards pass. Screen4/5; overall screen fails.
- Comments−0.32%, forward−1.21%, no-hate−0.000006469 absolute. Report every goal, including declines.
- Already explored cohort and overlapping training folds limit inference. This is retrospective reranker user-generalization evidence, not new-date confirmation or online uplift. Do not remove the follow guard or mark the goal complete just because aggregate evidence is positive.

## Next actions

1. Diagnosis complete: reports/emer-user-cv-tradeoffs-v1. All36 existing Jev candidates fail at least one inner guard;5/9 selected candidates already fail inner follow guard. Sparse-goal supervision mismatch is a hypothesis, not proven causality.
2. Implement and test the fixed18-fit experiment in configs/emer_balanced_transfer_v1.json and docs/EMER_BALANCED_TRANSFER.md. Both arms get identical task-balanced loss, anchor initialization and guarded inner selection. Freeze input/source hashes before execution. No training launched yet. Retain the original model code and outcomes. New results are exploratory because outer outcomes were already opened.
3. Seek actual improvement within the existing real-score resources and65 CNY cap. The positive aggregate result supports continued work but does not satisfy every registered requirement.

## Paths and preserved confirmation

Actual repo: /Users/hujiaoyuan/Documents/Codex/2026-10-08/referenced-chatgpt-conversation-this-is-an/outputs/ParetoJev.
Run data: /Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/user-cv-v1.
Logs: sibling user-cv-v1-train.log, user-cv-v1-evaluate.log and fold0/1/2 selection-audit logs.

The50K Apr23-25 cohort, input packet, original paired models and native producer preflight remain prepared but unscored due declined additional budget. Instance td89w6tsnx-ba43120e was confirmed OFF. No new paidGPU or externalJev inference was used in this run. Reviewable package: reports/multiobjective-confirmation-ready-2026-10-14/README.md.
