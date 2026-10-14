#!/usr/bin/env python3
"""Package only frozen, label-free confirmation prompts after source audit."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from audit_confirmation_sampling import audit_sampling
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash


def main(a):
    c=read_json(a.config);prep=read_json(a.data/'preparation.json');audit=read_json(a.data/'cohort_audit.json')
    if prep['config_hash']!=canonical_hash(c) or not audit['source_labels_features_and_histories_match'] or not audit['weights_verified']:
        raise ValueError('Frozen cohort source audit is required')
    if not audit.get('exact_prompt_alignment') or audit['raw_sha256']!=prep['raw_slates_sha256']:
        raise ValueError('Complete prompt and source-byte audit is required')
    development=a.data.parent/'emer-jev-scale-v3'
    prior_path=development/'prior-popularity.npy'
    if sha256_file(prior_path)!=read_json(development/'selection.json')['popularity_sha256']:
        raise ValueError('Frozen popularity prior changed')
    with np.load(a.data/'raw_slates.npz') as raw:
        prior=np.load(prior_path,mmap_mode='r')
        if not np.array_equal(raw['popularity'],prior[raw['items']]):
            raise ValueError('Candidate popularity differs from frozen prior')
    sampling=audit_sampling(a.data,c)
    source=a.data/'jev_input.jsonl';rows=0
    with source.open() as f:
        for line in f:
            r=json.loads(line)
            if set(r)!=set(('slate_id','split','candidate_ids','state','options','questions','objective_names')):
                raise ValueError('Unexpected outgoing fields')
            if r['slate_id']!=rows or r['split']!='test' or r['questions']!=c['questions'] or r['objective_names']!=c['objectives']:
                raise ValueError('Prompt protocol differs')
            state=json.loads(r['state'])
            if set(state)!=set(('task','context','recent_exposures_oldest_to_newest','candidates')):
                raise ValueError('Unexpected state fields')
            if state['context']['history_semantics']!='strictly earlier exposures; feedback is not supplied':
                raise ValueError('History semantics differ')
            if set(state['context'])!={'tab','history_semantics'}:
                raise ValueError('Unexpected context attributes')
            if len(r['options'])!=10 or len(r['candidate_ids'])!=10 or len(state['candidates'])!=10:
                raise ValueError('Candidate count differs')
            if any(set(item)!=set(('option','text','duration_seconds')) for item in state['candidates']):
                raise ValueError('Unexpected candidate attributes')
            rows+=1
    if rows!=50000:raise ValueError('Expected exactly50000 frozen slates')
    a.output.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as f,(a.output/'jev_input.jsonl.gz').open('wb') as target:
        with gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as compressed:
            for block in iter(lambda:f.read(1024*1024),b''):compressed.write(block)
    manifest=dict(slates=rows,planned_decisions=rows*7,independent_test_included=True,labels_included=False,
        source_input_sha256=sha256_file(source),compressed_sha256=sha256_file(a.output/'jev_input.jsonl.gz'),
        confirmation_config_sha256=sha256_file(a.config),development_registration_sha256=c['candidate_selection']['source_registration_sha256'],
        frozen_popularity_verified=True,popularity_sha256=sha256_file(prior_path),
        sampling_replay_sha256=sha256_file(a.data/'sampling_replay.json'),
        note='Input packaging only; no model outputs or feedback statistics used for tuning.')
    atomic_write_json(a.output/'transfer.json',manifest)
    print(manifest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_multiobjective_confirmation.json'))
    main(p.parse_args())
