#!/usr/bin/env python3
"""Fit the native-Jev arm on identical expanded cohorts; reuse audited baseline fits.

No independent test data is loaded. Public-service scores are never imported.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from paretojev.experiments.comparison import _output_lock,_source_identity
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import canonical_hash,sha256_file
from run_emer_jev import dataset,fit
from run_emer_scale import immutable_json


def native_arrays(folder,config,expected_input_sha):
    complete=read_json(folder/'complete.json');reg=read_json(folder/'registration.json')
    if not complete['complete'] or complete['public_cache_mixed'] or complete['registration_sha256']!=sha256_file(folder/'registration.json'):
        raise ValueError('Incomplete or mixed native collection')
    if reg['input_sha256']!=expected_input_sha or reg['test_opened'] or reg['slates']!=36000 or reg['goals']!=config['objectives']:
        raise ValueError('Native input protocol differs')
    if sha256_file(folder/'scores.npz')!=complete['scores_sha256'] or complete['valid_decisions']!=252000:
        raise ValueError('Native decision coverage or bytes differ')
    with np.load(folder/'scores.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    p=arrays['probabilities']
    if p.shape!=(36000,10,7) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid native distributions')
    if arrays['objective_names'].tolist()!=config['objectives'] or not np.array_equal(arrays['slate_ids'],np.arange(36000)):
        raise ValueError('Goal or row identities differ')
    source=reg['source']
    if str(arrays['model'])!=source['model'] or str(arrays['revision'])!=source['huggingface_revision']:
        raise ValueError('Model provenance differs')
    ent=np.clip(-(p*np.log(np.maximum(p,1e-300))).sum(1)/np.log(10),0,1)
    if not np.allclose(arrays['entropy'],ent[:,None,:],atol=1e-10) or not np.allclose(arrays['confidence'],1-arrays['entropy'],atol=1e-10):
        raise ValueError('Signal summary differs from actual categorical distribution')
    return arrays


def main(a):
    c=read_json(a.config);common=read_json(a.common/'common_manifest.json');baseline=read_json(a.common/'emer_selection.json')
    if common['identity']['config_hash']!=canonical_hash(c):raise ValueError('Expanded cohort configuration changed')
    for name,hash_value in common['files'].items():
        if sha256_file(a.common/name)!=hash_value:raise ValueError('Common data changed')
    if baseline['registration']['common_manifest_sha256']!=sha256_file(a.common/'common_manifest.json'):
        raise ValueError('Baseline was trained on another cohort')
    if baseline['registration']['source_sha256']!=_source_identity()['source_sha256'] or baseline['registration']['fit_script_sha256']!=sha256_file(Path('scripts/run_emer_jev.py')):
        raise ValueError('Model or fitting implementation changed after baseline training')
    for seed,r in baseline['selected'].items():
        if sha256_file(a.common/r['path'])!=r['checkpoint_sha256']:raise ValueError('Selected baseline changed')
    packet=read_json(a.transfer_manifest)
    if packet['labels_included'] or packet['independent_test_included']:raise ValueError('Unexpected transfer scope')
    scores=native_arrays(a.scores,c,packet['compressed_sha256']);data={}
    if str(scores['model'])!='autotrust/JEV-27B' or str(scores['revision'])!='14b425269ed24452e0a5ad5a8f3e51d139647a88':
        raise ValueError('Expected pinned AutoDL model')
    inputs=a.transfer_manifest.parent/'jev_input.jsonl.gz'
    if sha256_file(inputs)!=packet['compressed_sha256']:raise ValueError('Local prompt packet changed')
    with gzip.open(inputs,'rt') as f:rows=[json.loads(line) for line in f]
    seen=np.zeros((len(rows),7),dtype=bool)
    with sqlite3.connect('file:'+str((a.scores/'native.sqlite3').resolve())+'?mode=ro',uri=True) as db:
        group_count=0
        for i,key,text in db.execute('SELECT row_id,payload_hash,raw_json FROM groups'):
            if not 0<=i<len(rows):raise ValueError('Unexpected HTTP group')
            row=rows[i];wire=dict(kind='choice',state=row['state'],questions=row['questions'],options=row['options'],thinking='off')
            expected=hashlib.sha256(json.dumps(wire,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            group=json.loads(text)
            if key!=expected or group['question_count']!=7 or len(group['results'])!=7:raise ValueError('Original HTTP request/envelope differs')
            for j,child in enumerate(group['results']):
                saved=db.execute('SELECT raw_json FROM results WHERE row_id=? AND goal=?',(i,j)).fetchone()
                if saved is None or json.loads(saved[0])!=child:raise ValueError('Child differs from exact HTTP envelope')
            group_count+=1
        if group_count!=len(rows):raise ValueError('Missing original HTTP groups')
        for i,j,key,text in db.execute('SELECT row_id,goal,payload_hash,raw_json FROM results'):
            if not 0<=i<len(rows) or not 0<=j<7 or seen[i,j]:raise ValueError('Native response coverage differs')
            row=rows[i];p=dict(kind='choice',state=row['state'],question=row['questions'][j],options=row['options'],thinking='off')
            expected=hashlib.sha256(json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            r=json.loads(text)
            if key!=expected or r['model']!='autotrust/JEV-27B' or r['protocol']!='jev27-bare-v1' or r['options']!=row['options']:
                raise ValueError('Native response input/model alignment differs')
            if not np.array_equal(r['probabilities'],scores['probabilities'][i,:,j]):raise ValueError('Materialized score differs from native response')
            seen[i,j]=True
    if not seen.all():raise ValueError('Missing native responses')
    atomic_write_json(a.output/'native_response_audit.json',dict(decisions=int(seen.sum()),all_raw_responses_match=True,
        input_sha256=packet['compressed_sha256'],sqlite_sha256=sha256_file(a.scores/'native.sqlite3')))
    del rows,seen
    for name in c['split_names']:
        with np.load(a.common/f'{name}.npz',allow_pickle=False) as z:
            ids=z['slate_ids']
            if not np.array_equal(z['candidate_ids'],scores['candidate_ids'][ids]):raise ValueError('Candidate alignment changed')
        d=dataset(a.common/f'{name}.npz')
        for key,src in [('jev_scores','probabilities'),('jev_entropy','entropy'),('jev_confidence','confidence')]:
            d.arrays[key]=torch.from_numpy(scores[src][ids].astype('float32'))
        data[name]=d
    c=dict(c,model=baseline['registration']['model'])
    registration=dict(dataset_config_sha256=sha256_file(a.config),common_manifest_sha256=sha256_file(a.common/'common_manifest.json'),
        baseline_selection_sha256=sha256_file(a.common/'emer_selection.json'),native_complete_sha256=sha256_file(a.scores/'complete.json'),
        native_registration_sha256=sha256_file(a.scores/'registration.json'),package_sha256=_source_identity()['source_sha256'],
        script_sha256=sha256_file(Path(__file__)),fit_script_sha256=sha256_file(Path('scripts/run_emer_jev.py')),
        model=c['model'],training=c['training'],public_scores_used=False,independent_test_opened=False,
        baseline_reuse='Identical common inputs, architecture, fixed seeds, 4 trials per seed and 30 epochs; no extra baseline search.')
    immutable_json(a.output/'registration.json',registration);selected={};trials=[]
    for seed in c['training']['seeds']:
        choices=[]
        for i,(lr,coef) in enumerate((lr,coef) for lr in c['training']['learning_rates'] for coef in c['training']['prior_coefficients']):
            path=a.output/f'emer_native-seed{seed}-trial{i}.pt'
            if path.exists() and path.with_suffix('.json').exists():
                r=read_json(path.with_suffix('.json'))
                if sha256_file(path)!=r['checkpoint_sha256']:raise ValueError('Trial changed')
            else:_,r=fit(data,c,seed,True,lr,coef,path)
            r=dict(r,path=path.name);choices.append(r);trials.append(r)
            atomic_write_json(a.output/'progress.json',dict(completed_trials=len(trials),planned_trials=12,test_opened=False))
            print(seed,i,r['best_epoch'],r['validation'],flush=True)
        best=max(choices,key=lambda r:r['validation'])
        selected[str(seed)]={k:best[k] for k in ('path','checkpoint_sha256','validation','best_epoch','lr','prior_coefficient','parameters')}
    immutable_json(a.output/'selection.json',dict(registration=registration,selected=selected,trials=trials,
        baseline_selected=baseline['selected'],independent_test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('common','scores','output','transfer-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev_scale.json'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with _output_lock(a.output),threadpool_limits(limits=4):main(a)
