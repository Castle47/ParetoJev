#!/usr/bin/env python3
"""Fresh-process prediction and metric replay for the frozen v2 controls."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from run_kuairand_behavior_v2 import load_split, probabilities
from paretojev.training.behavior_calibration import apply_calibration
from paretojev.evaluation.behavior_metrics import metrics
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file, canonical_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=read_json(a.run/'results.json');manifest=read_json(a.data/'samples.json')
    if manifest!=r['samples']: raise ValueError('Sample manifest differs')
    checks={}
    with threadpool_limits(limits=r['config']['threads']):
        evaluation=load_split(a.data,manifest,'evaluation')
        for regime,evidence in r['models'].items():
            model_path=a.run/f'{regime}.joblib'
            if sha256_file(model_path)!=evidence['model_sha256']: raise ValueError('Model hash mismatch')
            payload=joblib.load(model_path)
            if payload['config_hash']!=canonical_hash(r['config']) or payload['sample_manifest_sha256']!=sha256_file(a.data/'samples.json'):
                raise ValueError('Model provenance mismatch')
            raw=probabilities(payload['models'],evaluation['x'],payload['columns'])
            calibrated=apply_calibration(raw,payload['calibration'])
            for kind,actual in [('raw',raw),('calibrated',calibrated)]:
                path=a.run/f'{regime}-{kind}.npy'
                if sha256_file(path)!=evidence[f'{kind}_sha256']: raise ValueError('Prediction changed')
                if not np.array_equal(actual,np.load(path)): raise ValueError('Prediction replay mismatch')
                if metrics(evaluation['y'],actual,evaluation['weight'])!=evidence[kind]:
                    raise ValueError('Metric replay mismatch')
            checks[regime]={'predictions_exact':True,'metrics_exact':True}
            print(f'{regime}: exact replay passed',flush=True)
    atomic_write_json(a.output,dict(status='passed',checks=checks,
        full_training_replayed=False,scope='fresh_process_same_machine',jev_requests=0))


if __name__=='__main__':main()
