"""Export complete or partial confirmation results with a consistent DB backup."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--archive-dir',type=Path,default=Path('/root'));a=p.parse_args()
stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
stage=a.output.parent/('confirmation-export-'+stamp);stage.mkdir()
for pattern in ('*.json','*.log','scores.npz'):
    for file in a.output.glob(pattern):shutil.copy2(file,stage/file.name)
source=a.output/'native.sqlite3'
if source.exists():
    with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src,sqlite3.connect(stage/'native.sqlite3') as dst:
        src.backup(dst)
        groups=dst.execute('SELECT COUNT(*) FROM groups').fetchone()[0]
        decisions=dst.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    registration=json.loads((stage/'registration.json').read_text()) if (stage/'registration.json').exists() else {}
    expected_groups=registration.get('slates',50000)
    expected_decisions=expected_groups*len(registration.get('goals',[None]*7))
    marker=json.loads((stage/'complete.json').read_text()) if (stage/'complete.json').exists() else {}
    complete=(groups==expected_groups and decisions==expected_decisions and marker.get('complete') is True
        and marker.get('http_groups')==groups and marker.get('valid_decisions')==decisions)
    (stage/'export_snapshot.json').write_text(json.dumps(dict(groups=groups,decisions=decisions,
        expected_groups=expected_groups,expected_decisions=expected_decisions,complete=complete,utc=stamp),indent=2)+'\n')
archive=a.archive_dir/('jev-confirmation-'+stamp+'.tgz')
with tarfile.open(archive,'w:gz') as t:
    for file in sorted(stage.iterdir()):
        if file.is_file():t.add(file,arcname='jev-confirmation-v1/'+file.name)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(archive=str(archive),sha256=digest)),flush=True)
