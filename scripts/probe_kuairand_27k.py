"""Read a bounded official archive prefix and inspect the complete small user table.

A probe is not full acquisition and cannot verify the published whole-archive MD5.
The archive's Python loader is never executed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import zlib

import pandas as pd
import requests

from paretojev.data.kuairand_27k import ASSETS, USER_FIELDS
from paretojev.utils.io import atomic_write_json


if __name__ == '__main__':
    limit=2*1024*1024;asset=ASSETS[0]
    with requests.get(asset['url'],headers={'Range':f'bytes=0-{limit-1}','Accept-Encoding':'identity'},
                      stream=True,timeout=(10,60)) as response:
        response.raise_for_status()
        if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes 0-{limit-1}/'):
            raise ValueError('Server must honor bounded range; no full download permitted by probe')
        compressed=response.raw.read(limit)
        if len(compressed)!=limit:raise ValueError('Incomplete prefix')
        content_range=response.headers['Content-Range']
    decoder=zlib.decompressobj(31)
    body=decoder.decompress(compressed,32*1024*1024)
    if decoder.unconsumed_tail:raise ValueError('Prefix expansion exceeds memory budget')
    content=None;members=[]
    with tarfile.open(fileobj=io.BytesIO(body),mode='r|') as archive:
        for member in archive:
            members.append({'name':member.name,'bytes':member.size})
            if member.name=='KuaiRand-27K/data/user_features_27k.csv':
                content=archive.extractfile(member).read()
                if len(content)!=member.size:raise ValueError('User table is not fully contained in the prefix')
                break
    if content is None:raise ValueError('No complete user table found in bounded prefix')
    root=Path('data/raw/kuairand-27k-probe');root.mkdir(parents=True,exist_ok=True)
    target=root/'user_features_27k.csv'
    if target.exists() and target.read_bytes()!=content:raise ValueError('Existing observed user table changed')
    target.write_bytes(content)
    frame=pd.read_csv(io.BytesIO(content))
    if set(USER_FIELDS)-set(frame.columns):raise ValueError('Published user schema changed')
    result={'scope':'complete_27k_user_table_from_bounded_official_archive_prefix','dataset':'KuaiRand-27K',
        'source_url':asset['url'],'http_range':content_range,'download_bytes':limit,
        'archive_md5_verified':False,'full_core_dataset_downloaded':False,
        'table_bytes':len(content),'table_sha256':hashlib.sha256(content).hexdigest(),
        'rows':len(frame),'distinct_user_ids':int(frame.user_id.nunique()),
        'user_id_min':int(frame.user_id.min()),'user_id_max':int(frame.user_id.max()),
        'raw_feature_count_excluding_id':len(frame.columns)-1,'snapshot_timestamp_present':False,
        'features':{name:{'dtype':str(frame[name].dtype),'missing':int(frame[name].isna().sum()),
                          'distinct_nonmissing':int(frame[name].nunique())} for name in frame.columns if name!='user_id'},
        'real_behavior_training_performed':False}
    path=Path('reports/kuairand-27k-readiness-2026-10-09/user_feature_probe.json')
    atomic_write_json(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='features'},indent=2))
