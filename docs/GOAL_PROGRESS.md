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

## Active fixed50/50 posterior experiment

The complete fallback diagnosis is in reports/emer-balanced-fallback-diagnosis. All45 seed29 Jev inner epochs improve the mean but fail at least one guard: long-view in folds0/1, comments in fold2. There was no missed admissible checkpoint. This motivates one fixed50/50 blend of original and fully task-balanced posterior, identical in both arms; no coefficient search or threshold changes.

- Config: configs/emer_blended_transfer_v1.json; protocol: docs/EMER_BLENDED_TRANSFER.md.
- Separate immutable driver: scripts/run_emer_blended_transfer.py. Old model and both prior driver bytes remain unchanged for exact replay.
- Five relevant tests pass; real source/anchor hashes verified and registration saved in reports/emer-blended-transfer-v1-start.
-18 fits, three original user folds, seeds7/17/29,15epochs, same budgets, initialization, inner guards and comparisons. Results remain exploratory after earlier outer inspections.

### Live handles — verify before polling/restarting

- Training session81273, PID39954: run_emer_blended_transfer.py --stage train.
- Finalizer session92872, PID39986: finish_emer_blended_transfer.sh waits for training, audits all18 fits, evaluates, replays and reports.
- caffeinate waits only for39986; not an automation.
- Output: external data root emer-jev-scale-v3/blended-transfer-v1. Logs: sibling blended-transfer-v1-train.log and blended-transfer-v1-evaluate.log.
- Stages prepare/train/audit/evaluate/replay. Keep driver, config and its imported frozen drivers unchanged while this registered run lives.

## Next actions

1. Poll the same live training session using60000ms empty waits. Do not duplicate it or read a partial outer result as completion.
2. Verify all18 checkpoint/inner-selection audit, then exact evaluation replay and all three comparisons. Report every seed, fold, goal and epoch0 fallback.
3. Keep the original criteria and65 CNY cap. Do not claim independent confirmation from these reused outer data. The frozen Apr23–25 queue remains unscored.

## Paths and preserved confirmation

Actual repo: /Users/hujiaoyuan/Documents/Codex/2026-10-08/referenced-chatgpt-conversation-this-is-an/outputs/ParetoJev.
Run data: /Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/emer-jev-scale-v3/user-cv-v1.
Logs: sibling user-cv-v1-train.log, user-cv-v1-evaluate.log and fold0/1/2 selection-audit logs.

The50K Apr23-25 cohort, input packet, original paired models and native producer preflight remain prepared but unscored due declined additional budget. Instance td89w6tsnx-ba43120e was confirmed OFF. No new paidGPU or externalJev inference was used in this run. Reviewable package: reports/multiobjective-confirmation-ready-2026-10-14/README.md.
