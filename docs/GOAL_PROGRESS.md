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

## Completed balanced-transfer follow-up

Report: reports/emer-balanced-transfer-v1/README.md. All18 fits, guarded inner-selection audit and independent prediction/metric replay pass. Training session20444 and finalizer88231 exited0; PIDs37992/38033 are finished. Do not restart them. Output: emer-jev-scale-v3/balanced-transfer-v1.

- Main controlled Jev increment: mean0.300586874 →0.305500498, +1.63%; absolute95% conditional interval[+0.003066959,+0.006823961].
- Compared with original EMER: +1.99%; balanced no-Jev training alone:+0.35%. Do not attribute the entire+1.99% to Jev.
- All point guards and positive-per-fold conditions pass. Follow+0.86%, comments+3.15%; long-view−0.73% remains a tradeoff.
- Seed29 is identical to its original anchor in every fold and both arms; its gain is exactly0. Strict positive-every-seed fails. Both required comparisons4/5; overall screen false.13/18 fits select epoch0.
- Post-outer-inspection exploratory design, not new independent confirmation. Do not drop seed29, change>0 to>=0, or mark achieved from only favorable averages.

## Completed fixed50/50 posterior experiment

Report: reports/emer-blended-transfer-v1/README.md. All18 fits, guarded inner-selection audit and exact independent prediction/metric replay pass. Training81273/PID39954 and finalizer92872/PID39986 exited0. Do not restart them. Output: emer-jev-scale-v3/blended-transfer-v1.

- Main matched-training Jev increment:+1.39%; absolute conditional95% interval[+0.001373,+0.007342]. Main screen2/5: seed29 negative, fold2 negative, long-view absolute−0.006784 exceeds0.005 drop guard.
- Reference against original EMER:+1.97%,5/5 reference conditions. This includes training changes: no-Jev mixed training alone adds+0.58%. Do not promote the reference success as controlled Jev success.
- All original criteria and both prior studies remain unchanged; all post-inspection studies remain exploratory. No paidGPU/API activity.
- Current evidence ledger: reports/emer-budget-summary-2026-10-15/README.md.

## Completed matched-comparator fallback policy

Report: reports/emer-matched-fallback-v1/README.md. Root protocol: configs/emer_matched_fallback_v1.json. All18 decisions across both prior recipes were frozen using inner metrics before new outer calculations. All checkpoint/decision audits and exact independent replay passed. No new training or inference requests.

- Balanced recipe unchanged:4/5, seed29 remains zero.
- Blended recipe retains5/9 Jev model slots;4 use their exact matched no-Jev model. This fixes two old-anchor fallback slots where the matched comparator had already improved.
- Main six-goal mean:0.301267045 →0.307199129, +1.97%, conditional absolute95% interval[+0.003325114,+0.009241747]. Main and original-reference comparisons both5/5.
- Original-EMER reference:+2.56%; no-Jev training change:+0.58%. Do not attribute total reference change solely to Jev.
- Main click−0.04%, long−0.66%, like+6.38%, follow+2.38%, comments+3.24%, forward+3.65%; no-hate absolute+0.000003430. Point guards pass; not every goal improves or has positive individual CI.
- Row-level delivery audit proves matched comparator predictions unchanged and every routed output exactly matches the frozen chosen source model. Both recipes, all seeds/folds and prior failures remain published.
- This is still post-inspection exploratory evidence. A5/5 screen does not prove new-date generalization or automatically complete the overall goal. All processes have exited successfully; there is no live training/evaluation session to poll.

## Next actions

1. Export complete: reports/emer-candidate-export-v1. Bundle at external development/exports/emer-matched-blended-v1, ZIP beside it. All9 branch/seed outputs on36K rows exactly match source scores/ranks/routes; standalone CLI outside repo and ZIP bytes verified. Manifest60a4b4b9fd9224c672e5b9bfc228ddce2e13afc4a250a6e3de1e793d9932de71. No live export/verification process remains.
2. Adapt independent-date evaluation readiness to these fold-specific encoders and models. The existing50K request packet is still unscored; do not feed full-development normalization into fold-specific models. User routing must use the original fixed hash, never outcomes.
3. Stop recipe/coefficient tuning against the explored outer cohort. Keep the65 CNY cap; no paidGPU or newJev requests. Stronger confirmation requires the matching real-score input/compute, which has not been provided. List the minimum missing input once the local handoff is complete; do not repeat the rejected budget request.

## Paths and preserved confirmation

Actual repo: /Users/hujiaoyuan/Documents/Codex/2026-10-08/referenced-chatgpt-conversation-this-is-an/outputs/ParetoJev.
Run data: /Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/user-cv-v1.
Logs: sibling user-cv-v1-train.log, user-cv-v1-evaluate.log and fold0/1/2 selection-audit logs.

The50K Apr23-25 cohort, input packet, original paired models and native producer preflight remain prepared but unscored due declined additional budget. Instance td89w6tsnx-ba43120e was confirmed OFF. No new paidGPU or externalJev inference was used in this run. Reviewable package: reports/multiobjective-confirmation-ready-2026-10-14/README.md.

## Candidate runtime files and remaining integration

- deployment/emer_candidate/candidate_runtime.py and infer.py: strict label-free numeric schema, original SHA256 user routing, per-fold encoding, immutable model/source hash checks, CPU2threads and batch32.
- scripts/export_emer_candidate.py requires a fresh export directory. scripts/verify_emer_candidate_export.py replays every existing36K prediction and standalone CLI.
- Do not modify the published bundle. Inference requires existing numeric behavior/Jev inputs; the upstream models are not embedded.
- Next integration can reuse audit_scores/materialize from scripts/evaluate_multiobjective_confirmation.py with its original scoring config unchanged. That helper produces old global features, which the new candidate must ignore: feed the audited raw26features to Candidate so its own per-fold specs are applied. Preserve the original50K request packet and producer schema. No test performance may be computed before every real score is present and audited.
