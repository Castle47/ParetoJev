#!/usr/bin/env python3
"""Independently map corrected signal tensors to cached native responses."""
import argparse
import json
from pathlib import Path

import numpy as np

from paretojev.jev.cache import DecisionCache
from paretojev.jev.public_choice import validate_choice
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file
from probe_jev_longview import QUESTION


def audit(a):
    c=read_json('configs/emer_jev.json')['jev'];checked={}
    for name,split,prompt,data_file in (
        ('train','train',a.core/'jev_input.jsonl',a.data/'dev-train.npz'),
        ('validation','validation',a.core/'jev_input.jsonl',a.data/'dev-validation.npz'),
        ('april22','test',a.data/'jev_input.jsonl',a.data/'test.npz')):
        folder=a.phase/('explicit-longview-'+name)
        reg=read_json(folder/'registration.json');record=read_json(folder/('result.json' if name=='validation' else 'complete.json'))
        input_key='inputs_sha256';data_key='validation_sha256' if name=='validation' else 'data_sha256'
        if reg['question']!=QUESTION or reg[input_key]!=sha256_file(prompt) or reg[data_key]!=sha256_file(data_file):
            raise ValueError('Question or source changed')
        if record['scores_sha256']!=sha256_file(folder/'scores.npz'):raise ValueError('Score bytes changed')
        rows=[json.loads(line) for line in prompt.read_text().splitlines()]
        rows=[r for r in rows if r['split']==split];z=np.load(folder/'scores.npz');original=np.load(data_file)
        expected=original['candidate_ids'] if name!='april22' else np.load(a.data/'raw_slates.npz')['items']
        np.testing.assert_array_equal(z['candidate_ids'],expected)
        if name=='april22' and reg['candidate_identity_sha256']!=sha256_file(a.data/'raw_slates.npz'):
            raise ValueError('Candidate source changed')
        np.testing.assert_array_equal(z['slate_ids'],np.array([r['slate_id'] for r in rows]))
        cache=DecisionCache(folder/'responses.sqlite3')
        for i,row in enumerate(rows):
            payload=dict(kind='choice',state=row['state'],question=QUESTION,options=row['options'])
            response=cache.get('explicit-longview',c['model']+'@'+c['revision'],payload)
            probs=validate_choice(response,model=c['model'],revision=c['revision'],options=row['options'])
            np.testing.assert_array_equal(z['probabilities'][i],probs)
        checked[name]=dict(real_decisions=len(rows),scores_sha256=record['scores_sha256'],
            original_inputs_and_labels_sha256=sha256_file(data_file),question_only_changed=True)
    atomic_write_json(a.phase/'corrected_signal_audit.json',dict(all_native_responses_match=True,splits=checked,
        total_real_decisions=sum(row['real_decisions'] for row in checked.values()),model=c['model'],revision=c['revision']))
    print('All corrected native responses and source mappings match',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--phase',type=Path,required=True);audit(p.parse_args())
