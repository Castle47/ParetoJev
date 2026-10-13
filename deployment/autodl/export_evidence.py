"""Export immutable probe evidence and a transactionally consistent score snapshot."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile

p=argparse.ArgumentParser();p.add_argument('--database',action='store_true');a=p.parse_args()
root=Path('/root/autodl-tmp');stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
stage=root/'jev-evidence'/stamp;stage.mkdir(parents=True)
for folder in sorted(root.glob('jev-native-v*')):
    dest=stage/folder.name;dest.mkdir()
    for pattern in ('*.json','*.log'):
        for file in folder.glob(pattern):shutil.copy2(file,dest/file.name)
service=stage/'service';service.mkdir()
for name in ('model-verified.json','source.json','fast-requirements.lock.txt'):
    path=root/'jev-service'/name
    if path.exists():shutil.copy2(path,service/name)
for name in ('torch_head_server.py','score_native.py','start_prefix_scoring.sh','model_manifest.json','transfer.json','transfer-verified.json'):
    path=Path('/root')/name
    if path.exists():shutil.copy2(path,service/name)
if a.database:
    source=root/'jev-native-v9-fp16/native.sqlite3';target=stage/'jev-native-v9-fp16/native.sqlite3'
    with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:
        src.backup(dst)
        counts=dict(groups=dst.execute('SELECT COUNT(*) FROM groups').fetchone()[0],
            decisions=dst.execute('SELECT COUNT(*) FROM results').fetchone()[0])
    counts.update(snapshot_utc=stamp,complete=counts['decisions']==252000 and counts['groups']==36000)
    (stage/'snapshot.json').write_text(json.dumps(counts,indent=2)+'\n')
archive=Path('/root')/('jev-evidence-'+stamp+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for file in sorted(stage.rglob('*')):
        if file.is_file():t.add(file,arcname=str(file.relative_to(stage)))
print(archive,flush=True)
