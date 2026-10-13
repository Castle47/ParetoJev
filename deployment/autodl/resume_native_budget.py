"""Resume unchanged inference after an explicitly authorized rental extension.

Keeps the original scientific registration intact. Changes only the recorded
wall-clock execution allowance, with a separate immutable receipt per resume.
"""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path

import score_native as native

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--stop-at',required=True);p.add_argument('--authorized-total-cny',type=float,required=True)
p.add_argument('--input',type=Path,default=Path('/root/jev_input.jsonl.gz'))
p.add_argument('--model-dir',type=Path,default=Path('/root/autodl-tmp/JEV-27B'))
a=p.parse_args();end=dt.datetime.fromisoformat(a.stop_at)
if end.tzinfo is None or end<=dt.datetime.now(dt.timezone.utc):raise ValueError('Future timezone-aware deadline required')
with (a.output/'.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reg=json.loads((a.output/'registration.json').read_text())
    if native.digest(native.__file__)!=reg['script_sha256'] or native.digest(a.input)!=reg['input_sha256']:
        raise ValueError('Cannot extend a changed scorer or input protocol')
    a.endpoint=reg['source']['endpoint'];a.workers=reg['workers'];a.max_attempts=reg['max_attempts']
    if native.producer(a.model_dir,a.endpoint)!=reg['source']:raise ValueError('Inference producer changed')
    if not json.loads((a.output/'probe.json').read_text())['passed']:raise ValueError('Probe has not passed')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    native.save(a.output/f'execution-extension-{stamp}.json',dict(original_registration_sha256=native.digest(a.output/'registration.json'),
        original_stop_at=reg['stop_at'],new_stop_at=a.stop_at,authorized_total_cny=a.authorized_total_cny,
        scientific_protocol_unchanged=True,note='Operator must obtain explicit user budget authorization before invoking this wrapper.'))
    native.collect(native.load_rows(a.input),a,a.output)
