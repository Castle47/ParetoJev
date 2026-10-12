#!/usr/bin/env python3
"""Continue this authorized experiment when its verified live scoring job finishes."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,runtime_provenance


def process_identity(pid):
    p=subprocess.run(['ps','-ww','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else ''


def finish(a):
    complete=a.data/'jev/complete.json';status=a.data/'finish_status.json'
    watch=process_identity(a.score_pid) if a.score_pid else ''
    if not complete.exists() and ('score_kuairand_core.py' not in watch or str(a.data) not in watch):
        raise ValueError('Scoring process identity is not live; inspect its original handle instead of guessing')
    paths=[Path('scripts')/name for name in ('audit_kuairand_core.py','materialize_kuairand_core.py',
        'run_kuairand_core.py','report_kuairand_core.py')]
    frozen={str(p):sha256_file(p) for p in paths}
    atomic_write_json(a.data/'finish_registration.json',dict(script_hashes=frozen,watched_pid=a.score_pid,
        watched_process=watch,provenance=runtime_provenance()))
    phase='waiting_for_live_scoring_job'
    atomic_write_json(status,dict(status=phase,watched_pid=a.score_pid,assistant_review_required=True))
    print('Waiting for verified scoring process',a.score_pid,flush=True)
    try:
        while not complete.exists():
            if process_identity(a.score_pid)!=watch:
                if complete.exists():break
                raise RuntimeError('Scoring process ended without complete evidence; inspect original scoring log/handle')
            time.sleep(30)
        source=read_json(complete)
        if not source['complete'] or sha256_file(a.data/'jev/scores.npz')!=source['scores_sha256']:
            raise ValueError('Scoring completion is not verified')
        cohort=read_json(a.report/'cohort_audit.json')
        if not cohort['all_source_occurrences_labels_and_weights_match']:raise ValueError('Real cohort audit missing')
        commands=[('audit_scores',[sys.executable,'scripts/audit_kuairand_core.py','--mode','scores','--data',str(a.data),
                    '--scores',str(a.data/'jev'),'--output',str(a.report/'score_audit.json')])]
        if not (a.data/'full/manifest.json').exists():
            commands.append(('materialize_real_scores',[sys.executable,'scripts/materialize_kuairand_core.py',
                '--data',str(a.data),'--scores',str(a.data/'jev'),'--tree',str(a.tree),'--output',str(a.data/'full')]))
        commands += [('fit_and_evaluate',[sys.executable,'scripts/run_kuairand_core.py','--data',str(a.data/'full'),
                      '--output',str(a.data/'comparison'),'--stage','full']),
                     ('replay',[sys.executable,'scripts/run_kuairand_core.py','--data',str(a.data/'full'),
                      '--output',str(a.data/'comparison'),'--stage','replay']),
                     ('report',[sys.executable,'scripts/report_kuairand_core.py','--data',str(a.data),
                      '--run',str(a.data/'comparison'),'--output',str(a.report)])]
        for phase,command in commands:
            if any(sha256_file(path)!=digest for path,digest in frozen.items()):raise ValueError('Dependent script changed while waiting')
            atomic_write_json(status,dict(status='running',phase=phase,command=command,assistant_review_required=True))
            print('Starting',phase,flush=True)
            with (a.data/f'finish-{phase}.log').open('a') as handle:
                handle.write('\nCOMMAND '+json.dumps(command)+'\n');handle.flush()
                subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
        atomic_write_json(status,dict(status='complete',report=str(a.report/'README.md'),assistant_review_required=True))
        print('All core artifacts ready for independent assistant review',flush=True)
    except BaseException as exc:
        atomic_write_json(status,dict(status='stopped_with_error',phase=phase,error_type=type(exc).__name__,
            error=str(exc),assistant_review_required=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','tree','report'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--score-pid',type=int);a=p.parse_args()
    with (a.data/'.finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);finish(a)
