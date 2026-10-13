#!/usr/bin/env python3
"""Complete real scoring, audit, confirmation, replay and reporting in order."""
import argparse
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

from paretojev.utils.io import atomic_write_json
from paretojev.utils.provenance import sha256_file
from paretojev.experiments.comparison import _output_lock


def finish(a):
    if not (a.data/'preparation.json').exists() or not (a.data/'selection.json').exists():
        raise ValueError('Freeze fresh cohort and fitted arms first')
    atomic_write_json(a.data/'execution_registration.json',dict(
        config_sha256=sha256_file(a.config),selection_sha256=sha256_file(a.data/'selection.json'),
        raw_slates_sha256=sha256_file(a.data/'raw_slates.npz'),
        scripts={p:sha256_file(Path('scripts')/p) for p in
            ('prepare_emer_jev.py','run_emer_jev.py','score_emer_jev.py','score_kuairand_core.py',
             'audit_emer_jev.py','report_emer_jev.py','finish_emer_jev.py')}))
    common=['--root',str(a.root),'--output',str(a.data),'--config',str(a.config)]
    audit=['--root',str(a.root),'--data',str(a.data),'--config',str(a.config)]
    stages=[('score',['scripts/score_emer_jev.py','--data',str(a.data),'--config',str(a.config)]),
        ('audit',['scripts/audit_emer_jev.py',*audit]),
        ('confirm',['scripts/run_emer_jev.py',*common,'--stage','confirm']),
        ('replay',['scripts/run_emer_jev.py',*common,'--stage','replay']),
        ('metric_audit',['scripts/audit_emer_jev.py',*audit,'--metrics-only']),
        ('report',['scripts/report_emer_jev.py','--data',str(a.data),'--output',str(a.report)])]
    completed=[]
    for name,command in stages:
        status=dict(stage=name,completed=completed,status='running',utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(a.data/'finish_status.json',status);print('Stage:',name,flush=True)
        with (a.data/f'{name}.log').open('a') as log:
            result=subprocess.run([sys.executable,'-u',*command],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_write_json(a.data/'finish_status.json',dict(status, status='failed',exit_code=result.returncode))
            raise RuntimeError(f'{name} failed; inspect {a.data/name}.log; no replacement or test tuning')
        completed.append(name)
    atomic_write_json(a.data/'finish_status.json',dict(stage='complete',completed=completed,status='complete',
        utc=datetime.now(timezone.utc).isoformat()))
    print('Two-arm experiment, audit and replay complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/emer_jev.json'))
    a=p.parse_args()
    with _output_lock(a.data/'finisher'):finish(a)
