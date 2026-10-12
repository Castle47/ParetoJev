# Core experiment is running — no effectiveness conclusion yet

Latest goal: prioritize real ParetoJev vs behavior-only EMER-like/CMR-like, with
same-Jev and no-Jev controls. Do not return to upstream SASRec or rare-feedback tuning.

Authoritative experiment root: `/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k/core-choice-v1`.
Configuration: `configs/kuairand_core.json`. Six arms × three seeds; 20 fixed epochs.
The full 2,000-slate/20,000-candidate cohort and all source records/labels/weights
have passed independent audit. Nine no-Jev checkpoints are frozen. No test model
comparison has run. 260 tests and a staged synthetic training/evaluation/replay
passed; synthetic results are engineering-only.

## Live work (verify liveness before acting)

- Real Choice scorer: unified exec session `53190`, process `40712`.
- Automatic dependent-work driver: session `89698`, process `41371`.
- Scoring journal: `jev/progress.json`, `jev/attempts.jsonl`, `jev/responses.sqlite3`.
- Controller state: `finish_status.json`; controller stdout in
  `../../work/kuairand-core-finish.log` relative to the repo.
- Scorer stdout in `../../work/kuairand-core-jev.log`.

The driver verifies the scoring process identity. After complete scores exist it
runs raw-response audit, materializes real-score tensors, fits remaining Jev arms,
evaluates all six arms plus a fixed behavior control, replays all predictions and
metrics, and generates the report. It stops with a recorded error if a dependency
fails. Do not start a duplicate scorer/driver merely because a poll times out.
Poll the existing handles (60 seconds) or verify the exact PID/command. A live
job is a verified wait, not an external-input blocker.

Package sources and the core runner are fingerprinted by the nine frozen models;
the driver also fingerprints its dependent scripts. Do not edit those numerical
paths while this registered experiment is running. Documentation edits are safe.
If a real defect requires a change, preserve the interrupted run and explicitly
register the repair; do not silently mix changed code with old evidence.

## Completion review still required

`finish_status.json: complete` only means artifacts are ready for review. Inspect
all 18 checkpoints/receipts, real scores and source audits, common-input hashes,
three-seed primary/equal-signal results, unchanged gates, all confidence intervals
and exact replay before answering the research question or completing the goal.
EMER-like/CMR-like remain project variants. Weighted retrospective pseudo-slates
are not causal online experiments. A failed or inconclusive result must be reported
as such; no threshold lowering or easiest-comparator selection.
