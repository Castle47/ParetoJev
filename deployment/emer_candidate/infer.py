#!/usr/bin/env python3
"""Run frozen reranker inference from supplied numeric inputs; no API calls."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'python'))
from candidate_runtime import Candidate
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['emer','emer_jev','anchor'],default='emer_jev');p.add_argument('--seed',type=int,choices=[7,17,29],default=7)
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    model=Candidate(root);result=model.predict(inputs,arm=a.arm,seed=a.seed)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**result)
    print(json.dumps(dict(slates=len(result['scores']),arm=a.arm,seed=a.seed,output=str(a.output),jev_slates=int(result['uses_jev'].sum()))))
