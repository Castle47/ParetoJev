"""Export ignored files required for offline replay alongside a source Git bundle."""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

import run_microlens_confirmatory as run
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file


def main():
    registration = run.verify()
    if read_json(run.REPORT/'replay.json')['status'] != 'passed':
        raise ValueError('Require successful offline replay before export')
    if read_json(run.REPORT/'collection_replay.json')['http_calls'] != 0:
        raise ValueError('Require a zero-HTTP collection replay')
    paths = {Path(name) for name in registration['input_hashes'] | registration['prepared_hashes']}
    paths.update(run.ROOT/name for name in ('sasrec.pt','frozen_predictions.npz','validation_predictions.npy',
        'scores.jsonl','requests.sqlite3','http_attempts.jsonl','errors.jsonl','collection_identity.json',
        'scoring_report.json','per_user_metrics.json'))
    paths.update((run.ROOT/'collection_runs').glob('*.json'))
    tracked = set(subprocess.check_output(['git','ls-files','-z'],text=True).split('\0'))
    root = Path.cwd().resolve()
    entries = []
    for path in sorted(paths):
        if path.is_absolute() or '..' in path.parts or path.is_symlink() or not path.is_file():
            raise ValueError(f'Unsafe or missing replay input: {path}')
        path.resolve().relative_to(root)
        if str(path) not in tracked:
            entries.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file(path)})
    output = Path('../ParetoJev-confirmatory-replay-2026-10-09.tar.gz')
    if output.exists(): raise FileExistsError(output)
    manifest = {'scope':'offline_checkpoint_collection_and_evaluation_replay',
        'source_commit_at_export':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'registration_sha256':sha256_file(run.REPORT/'registration.json'),
        'research_source_hashes':registration['source_hashes'], 'files':entries,
        'note':'Extract into a matching source checkout. No virtual environment or credentials are included.'}
    encoded = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    with output.open('wb') as dest, gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            info=tarfile.TarInfo('REPLAY_DATA_MANIFEST.json');info.size=len(encoded);info.mode=0o644
            archive.addfile(info,io.BytesIO(encoded))
            for entry in entries:
                path=Path(entry['path']);info=tarfile.TarInfo(entry['path']);info.size=entry['bytes'];info.mode=0o644
                with path.open('rb') as handle:archive.addfile(info,handle)
    result={**manifest,'archive':output.name,'archive_bytes':output.stat().st_size,'archive_sha256':sha256_file(output)}
    atomic_write_json(output.with_name('ParetoJev-confirmatory-replay-2026-10-09.manifest.json'),result)
    print(json.dumps({'files':len(entries),'archive_bytes':result['archive_bytes'],'archive_sha256':result['archive_sha256']},indent=2))


if __name__=='__main__':
    main()
