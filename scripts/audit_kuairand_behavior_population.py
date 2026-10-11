#!/usr/bin/env python3
"""Descriptive population-label audit and source-offset index; no model fitting."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_27k import BINARY_LABELS
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/kuairand_27k_behavior_v2.json'))
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root=args.root;v2=root/'behavior-v2'
    c=read_json(args.config);source=read_json(root/'behavior-pilot-v1/events.json')
    if source['dtype'] != json.loads(json.dumps(EVENT.descr)):
        raise ValueError('Event-store dtype does not match this reader')
    samples=read_json(v2/'samples.json');out=v2/'rare-audit';out.mkdir(exist_ok=True)
    names=('train','calibration','evaluation');bounds=[0,c['train_end_ms'],c['calibration_end_ms'],c['evaluation_end_ms']]
    counts=np.zeros(3,dtype='int64');positives=np.zeros((3,7),dtype='int64');parts={s:[] for s in names}
    h=hashlib.sha256();offset=0
    with (root/'behavior-pilot-v1/events.bin').open('rb') as f:
     while raw:=f.read(EVENT.itemsize*1000000):
      h.update(raw);e=np.frombuffer(raw,dtype=EVENT);labels=np.unpackbits(e['labels'][:,None],axis=1,bitorder='little')[:,:7]
      for j,name in enumerate(names):
       mask=(e['time']>=bounds[j])&(e['time']<bounds[j+1]);counts[j]+=int(mask.sum())
       positives[j]+=labels[mask].sum(axis=0,dtype='int64')
       parts[name].append(offset+np.flatnonzero(mask & ((e['labels']&64)!=0)))
      offset+=len(e)
    if h.hexdigest()!=source['store_sha256'] or offset!=source['rows']: raise ValueError('Source mismatch')
    result={'scope':'post-experiment descriptive full-period label audit; no model or gate changed',
     'source_store_sha256':h.hexdigest(),'index_unit':'zero_based_EVENT_records','event_itemsize':EVENT.itemsize,'objective_order':list(BINARY_LABELS),'periods':{}}
    for j,name in enumerate(names):
     assert counts[j]==samples['splits'][name]['represented_exposures']
     index=np.concatenate(parts[name]);path=out/f'{name}-hate-positive-offsets.npy';np.save(path,index)
     result['periods'][name]={'exposures':int(counts[j]),'positive_counts':positives[j].tolist(),
      'prevalence':(positives[j]/counts[j]).tolist(),'hate_positive_index_rows':len(index),'hate_positive_index_sha256':sha256_file(path)}
    atomic_write_json(args.report,result)
    print(result,flush=True)


if __name__ == '__main__':
    main()
