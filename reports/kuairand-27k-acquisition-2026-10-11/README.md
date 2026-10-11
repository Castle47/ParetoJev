# KuaiRand-27K acquisition on external storage

The user supplied `/Volumes/曾诚的移动硬盘/`. Write, fsync, atomic rename and
exclusive file locking passed. The selected data root is
`/Volumes/曾诚的移动硬盘/ParetoJev/kuairand-27k`.
About 883 GiB was available at startup; the previous storage blocker is resolved.
Existing files elsewhere on the disk were not changed.

The active job downloads the checksum-pinned full 27K archive and both content
supplements. Complete download, extraction, dataset audit and behavior training
are separate milestones. A `.part` file is not a verified finished download.
No recommendation benchmark or trained 27K model is claimed by this report.

See `storage_check.json` and `acquisition_plan.json` for the initial checks.
The external data root holds persistent partial files and acquisition manifests.

Two long-stream category transfers hit read timeouts. The verified core archive
and captions were retained. Categories now resume using exact 32 MiB HTTP byte
ranges with at most eight transport retries per invocation. TLS verification
and the final whole-file published MD5 remain mandatory; HTTP errors are not
automatically retried. Tests verify that partial-range recovery neither loses
nor duplicates bytes and refuses a server that ignores requested ranges.

Current engineering checks: 229 tests passed in 15.15 seconds. This is not a
recommendation benchmark. Completion is reported only after file checksums and
the separate audit stage finish.
