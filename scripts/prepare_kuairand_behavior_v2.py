#!/usr/bin/env python3
import argparse
import fcntl
from pathlib import Path
from paretojev.data.kuairand_features_v2 import make_metadata, prepare
from paretojev.utils.io import read_json, atomic_write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/kuairand_27k_behavior_v2.json'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=read_json(a.config)
        if (a.output/'registration.json').exists() and read_json(a.output/'registration.json')!=c:
            raise ValueError('Registration differs')
        atomic_write_json(a.output/'registration.json',c)
        metadata,_=make_metadata(a.root/'extracted/KuaiRand-27K/data/video_features_basic_27k.csv',
            a.root/'extracted/extraction_manifest.json',a.cache,a.output)
        prepare(a.root/'behavior-pilot-v1',metadata,a.output,c)
        print('Preparation complete',flush=True)


if __name__=='__main__':main()
