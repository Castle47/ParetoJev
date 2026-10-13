#!/usr/bin/env python3
"""Export bounded-correction replay inputs and actual native response evidence."""
import argparse
import gzip
import sqlite3
import tarfile
import tempfile
from pathlib import Path

from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file


def export(a):
    if read_json(a.phase/'replay.json')['status']!='exact':raise ValueError('Verified replay required')
    files={}
    for path in sorted(a.phase.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.npz'):
            files['emer-jev-v2/'+str(path.relative_to(a.phase))]=path
    for name in ('selection.json','test.npz','dev-train.npz','dev-validation.npz'):
        files['emer-jev-v1/'+name]=a.data/name
    for arms in read_json(a.data/'selection.json')['selected'].values():
        name=arms['emer']['path'];files['emer-jev-v1/'+name]=a.data/name
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='jev-signal-export-') as tmp:
        for i,source in enumerate(sorted(a.phase.rglob('responses.sqlite3'))):
            target=Path(tmp)/f'cache-{i}.sqlite3'
            with sqlite3.connect(source) as src,sqlite3.connect(target) as dst:src.backup(dst)
            files['emer-jev-v2/'+str(source.relative_to(a.phase))]=target
        manifest=Path(tmp)/'FILES.json'
        atomic_write_json(manifest,dict(purpose='saved-coefficient prediction/metric replay and native response evidence; full source cohort audits require original logs',
            files={name:dict(sha256=sha256_file(path),bytes=path.stat().st_size) for name,path in sorted(files.items())}))
        files['FILES.json']=manifest;partial=a.output.with_name(a.output.name+'.partial')
        with (partial.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,
            tarfile.open(fileobj=gz,mode='w') as archive):
            for name,path in sorted(files.items()):
                info=archive.gettarinfo(str(path),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as handle:archive.addfile(info,handle)
        partial.replace(a.output)
    digest=sha256_file(a.output);a.output.with_name(a.output.name+'.sha256').write_text(digest+'  '+a.output.name+'\n')
    print(dict(files=len(files),bytes=a.output.stat().st_size,sha256=digest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--phase',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);export(p.parse_args())
