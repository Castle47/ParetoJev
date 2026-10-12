#!/usr/bin/env python3
"""Export frozen evaluation evidence; intentionally excludes full training data."""
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from paretojev.utils.io import read_json
from paretojev.utils.provenance import sha256_file


def export(a):
    receipt = read_json(a.run/'replay.json')
    if not receipt['exact_predictions_metrics_and_bootstrap'] or receipt['results_sha256'] != sha256_file(a.run/'results.json'):
        raise ValueError('Verified replay required')
    if a.output.exists(): raise ValueError('Existing archive')
    with tempfile.TemporaryDirectory(prefix='aligned-export-', dir=a.output.parent) as temp:
        stage = Path(temp)
        def copy(root, dest, names):
            (stage/dest).mkdir(exist_ok=True)
            for name in names: shutil.copyfile(root/name, stage/dest/name)
        copy(a.data, 'data', ['samples.json']+[f'evaluation-{key}.npy' for key in ('x','y','weight','user','time')])
        copy(a.sequences, 'sequences', ['sequences.json']+[f'evaluation-{key}.npy' for key in ('history','item','record','history_max_time')])
        copy(a.tree, 'tree', ['results.json','upload_metadata_assumed_static.joblib',
            'upload_metadata_assumed_static-raw.npy','upload_metadata_assumed_static-calibrated.npy'])
        names=['results.json','replay.json','frozen_models.json','registration.json','resume_provenance.json']
        names += [f'{regime}.pt' for regime in ('tabular_mmoe','sasrec_mmoe')]
        names += [f'{regime}-{mode}.npy' for regime in ('tree','tabular_mmoe','sasrec_mmoe') for mode in ('raw','calibrated')]
        copy(a.run, 'run', [name for name in names if (a.run/name).exists()])
        (stage/'README.txt').write_text('Frozen KuaiRand-27K aligned development evaluation only. No full training data.\n'
            'Use the accompanying repository/environment, from its root:\n'
            '.venv/bin/python scripts/run_kuairand_aligned.py --data PACKAGE/data --sequences PACKAGE/sequences '
            '--tree PACKAGE/tree --output PACKAGE/run --replay\n'
            'Evaluation includes previously seen April 19 data; not a confirmatory or Jev/ParetoJev benchmark.\n')
        with tarfile.open(a.output,'w:gz') as archive:
            for path in sorted(stage.iterdir()): archive.add(path,arcname=path.name)
    a.output.with_suffix(a.output.suffix+'.sha256').write_text(f'{sha256_file(a.output)}  {a.output.name}\n')
    print(f'Exported {a.output}: {a.output.stat().st_size} bytes')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('run','data','sequences','tree','output'): p.add_argument('--'+name,type=Path,required=True)
    export(p.parse_args())
