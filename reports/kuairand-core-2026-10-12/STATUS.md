# Core validation completed — overall superiority not supported

The real scorer, dependent-work driver, all 18 training runs, response audit,
metric replay and fresh-directory replay are complete. No experiment job needs
restarting. Historical `progress_snapshot.json` is a checkpoint, not current status.

Read `README.md` for the final conclusion; `REGISTERED_MONTE_CARLO_REPORT.md` and
`summary.json` retain the untouched original numerical run. Independent exact-HV
refinement and all mathematical/artifact checks are in `independent_verification.json`.
`completion_audit.json` maps the current validation requirements to evidence.

Primary point gates: 1/4. The exact mean of per-seed HV gains is +9.46%, but varies
+59.79% / -35.84% / +4.43%; the supplementary ratio of absolute HV means is +0.27%.
Cold gain, regret and controllability fail their original primary thresholds.
Stronger no-Jev/equal-Jev controls prevent an overall superiority claim.

The current validation question is answered by this negative/unstable result.
The project's aspirational research/serving targets remain unachieved. Any future
optimization should use a newly registered protocol and preserve this evidence.
