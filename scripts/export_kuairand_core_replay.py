#!/usr/bin/env python3
"""Portable frozen core experiment, including real response evidence, not raw 27K logs."""
import argparse
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def export(a):
    if a.output.exists():raise ValueError('Existing archive; choose a fresh destination')
    replay=read_json(a.data/'comparison/replay.json')
    if not replay['exact_predictions_metrics_and_intervals'] or replay['summary_sha256']!=sha256_file(a.data/'comparison/summary.json'):
        raise ValueError('Verified frozen comparison required')
    with tempfile.TemporaryDirectory(prefix='core-replay-',dir=a.output.parent) as temporary:
        stage=Path(temporary)
        for folder in ('base','full'):
            shutil.copytree(a.data/folder,stage/folder)
        shutil.copytree(a.data/'comparison',stage/'comparison',ignore=shutil.ignore_patterns('replay-predictions.npz'))
        (stage/'jev').mkdir()
        for name in ('registration.json','attempts.jsonl','complete.json','scores.npz'):
            shutil.copyfile(a.data/'jev'/name,stage/'jev'/name)
        # Backup produces a consistent SQLite snapshot, independent of WAL state.
        with sqlite3.connect(f'file:{(a.data/"jev/responses.sqlite3").resolve()}?mode=ro',uri=True) as source:
            with sqlite3.connect(stage/'jev/responses.sqlite3') as target:source.backup(target)
        for name in ('raw_slates.npz','selection.npz','selection.json','features.json','captions.json',
                     'prompts.json','jev_input.jsonl','selected_captions.json','preparation.json'):
            shutil.copyfile(a.data/name,stage/name)
        (stage/'audits').mkdir()
        for name in ('cohort_audit.json','score_audit.json','independent_verification.json','replay.json'):
            shutil.copyfile(a.report/name,stage/'audits'/name)
        (stage/'README.txt').write_text('Frozen KuaiRand-27K core comparison evidence. No full raw logs or upstream behavior training arrays.\n'
            'Use the accompanying source bundle and frozen Python environment, from repository root:\n'
            '.venv/bin/python scripts/run_kuairand_core.py --data PACKAGE/full --output PACKAGE/comparison --stage replay\n'
            '.venv/bin/python scripts/audit_kuairand_core.py --mode scores --data PACKAGE --scores PACKAGE/jev --output PACKAGE/raw-score-replay.json\n'
            'Optional exact-HV verification uses requirements-hv-audit.txt in an isolated --target directory.\n'
            'No network requests are needed for these replays. Original logs are required to rerun the source-occurrence audit.\n'
            'The registered Monte Carlo summary is untouched. audits/independent_verification.json contains the exact-HV refinement.\n'
            'Final interpretation is in the accompanying source repository reports/kuairand-core-2026-10-12/README.md.\n')
        manifest={str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
                  for p in sorted(stage.rglob('*')) if p.is_file()}
        atomic_write_json(stage/'FILES.json',manifest)
        with tarfile.open(a.output,'w:gz') as archive:
            for path in sorted(stage.iterdir()):archive.add(path,arcname=path.name)
    digest=sha256_file(a.output)
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(f'{digest}  {a.output.name}\n')
    atomic_write_json(a.report/'export.json',dict(file=a.output.name,bytes=a.output.stat().st_size,sha256=digest,
        scope='frozen reranker tensors/checkpoints/predictions/raw real Jev response cache; no complete raw 27K logs'))
    print('Exported',a.output,a.output.stat().st_size,'bytes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','report','output'):p.add_argument('--'+key,type=Path,required=True)
    export(p.parse_args())
