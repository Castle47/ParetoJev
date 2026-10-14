#!/usr/bin/env python3
"""Verify portable runtime against every frozen36K routed prediction."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deployment/emer_candidate'))
from candidate_runtime import Candidate,sha
from export_emer_candidate import inference_inputs
from run_emer_user_cv import common_arrays,roles_for_fold
from paretojev.utils.io import read_json,atomic_write_json


def verify(a):
    model=Candidate(a.bundle);common,raw=common_arrays(SimpleNamespace(data=a.development));inputs=inference_inputs(common,raw)
    original=read_json(a.development/'user-cv-v1/registration.json')['config'];records=[]
    source=a.development/'matched-fallback-v1/blended'
    with np.load(source/'predictions.npz') as z:expected={k:z[k] for k in z.files}
    for seed in model.manifest['seeds']:
        for arm in ('emer','emer_jev','anchor'):
            result=model.predict(inputs,arm=arm,seed=seed)
            if not np.array_equal(result['scores'],expected[f'{arm}/{seed}']):raise ValueError(f'Export prediction differs: {arm}/{seed}')
            for f in range(3):
                if not np.array_equal(result['fold_ids']==f,roles_for_fold(common['user_ids'],f,original)==2):raise ValueError('User routing differs')
            order=np.argsort(-expected[f'{arm}/{seed}'],axis=1,kind='stable')
            if not np.array_equal(result['ranking_indices'],order) or not np.array_equal(result['ranked_candidate_ids'],np.take_along_axis(common['candidate_ids'],order,axis=1)):raise ValueError('Output ranking differs')
            records.append(dict(arm=arm,seed=seed,slates=len(raw),predictions_exact=True,ranking_exact=True,routing_exact=True))
            print('Export replay exact:',arm,seed,len(raw),flush=True)
    with tempfile.TemporaryDirectory() as directory:
        output=Path(directory)/'prediction.npz'
        command=[sys.executable,str(a.bundle.resolve()/'infer.py'),'--input',str(a.bundle.resolve()/'example_input.npz'),'--output',str(output),'--arm','emer_jev','--seed','7']
        subprocess.run(command,cwd=directory,check=True,capture_output=True,text=True)
        with np.load(a.bundle/'example_input.npz') as z:example={k:z[k] for k in z.files}
        replay=model.predict(example)
        with np.load(output) as z:
            if set(z.files)!=set(replay) or any(not np.array_equal(z[k],v) for k,v in replay.items()):raise ValueError('Standalone CLI differs')
    receipt=dict(manifest_sha256=sha(a.bundle/'manifest.json'),archive_sha256=sha(a.bundle.with_suffix('.zip')),
        source_predictions_sha256=sha(source/'predictions.npz'),checks=records,standalone_cli_exact=True,
        label_free_input=True,independent_test_read=False,benchmark_scope='Exact inference replay on the already-explored development cohort only')
    atomic_write_json(a.output,receipt);print('All exported predictions and standalone CLI verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);verify(p.parse_args())
