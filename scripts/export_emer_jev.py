#!/usr/bin/env python3
"""Export checkpoint replay inputs; full source logs stay on the external disk."""
import argparse
import gzip
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def export(a):
    if not read_json(a.data/'replay.json')['exact_predictions_and_metrics']:
        raise ValueError('Verified replay required')
    files={}
    for path in sorted(a.data.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.npz','.pt'):
            files['emer-jev-v1/'+str(path.relative_to(a.data))]=path
    for name in ('upload_metadata_assumed_static.joblib','results.json'):
        files['behavior-v2/run-v2/'+name]=a.root/'behavior-v2/run-v2'/name
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='emer-export-') as temp:
        cache=Path(temp)/'responses.sqlite3'
        with sqlite3.connect(a.data/'jev/responses.sqlite3') as source,sqlite3.connect(cache) as target:
            source.backup(target)
        files['emer-jev-v1/jev/responses.sqlite3']=cache
        manifest=dict(purpose='prediction/metric replay plus raw Jev evidence; full history/source audits require original logs',
            files={name:dict(sha256=sha256_file(path),bytes=path.stat().st_size) for name,path in sorted(files.items())})
        manifest_path=Path(temp)/'FILES.json';atomic_write_json(manifest_path,manifest);files['FILES.json']=manifest_path
        temporary=a.output.with_name(a.output.name+'.partial')
        with (temporary.open('wb') as raw,
              gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed,
              tarfile.open(fileobj=compressed,mode='w') as archive):
            for name,path in sorted(files.items()):
                info=archive.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as handle:archive.addfile(info,handle)
        temporary.replace(a.output)
    digest=sha256_file(a.output)
    a.output.with_name(a.output.name+'.sha256').write_text(digest+'  '+a.output.name+'\n')
    print(json.dumps(dict(path=str(a.output),bytes=a.output.stat().st_size,sha256=digest,files=len(files)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);export(p.parse_args())
