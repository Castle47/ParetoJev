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

## Next actions

1. Inspect seed29 inner histories to determine why no admissible improving epoch was found: guard violations, no aggregate gain, or a training issue. Do not select from outer performance or rewrite past outcomes.
2. Decide whether evidence supports a concrete local improvement hypothesis. Any changed recipe must have a new exploratory registration and matched no-Jev control; do not keep trying settings simply to make known outer metrics pass.
3. Preserve the65 CNY cap and the unscored independent confirmation. Even a favorable exploratory result does not remove the need for stronger confirmation before broad claims.

## Paths and preserved confirmation

Actual repo: /Users/hujiaoyuan/Documents/Codex/2026-10-08/referenced-chatgpt-conversation-this-is-an/outputs/ParetoJev.
Run data: /Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/user-cv-v1.
Logs: sibling user-cv-v1-train.log, user-cv-v1-evaluate.log and fold0/1/2 selection-audit logs.

The50K Apr23-25 cohort, input packet, original paired models and native producer preflight remain prepared but unscored due declined additional budget. Instance td89w6tsnx-ba43120e was confirmed OFF. No new paidGPU or externalJev inference was used in this run. Reviewable package: reports/multiobjective-confirmation-ready-2026-10-14/README.md.
