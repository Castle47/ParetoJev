"""Small, credential-free status report for the dedicated AutoDL instance."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

root=Path('/root/autodl-tmp');model=root/'JEV-27B'
report={'time':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,path in [('system','/root'),('data',str(root))]:
    disk=shutil.disk_usage(path);report[name+'_free_gib']=round(disk.free/2**30,2)
report['finished_model_shards']=len(list(model.glob('model-*.safetensors')))
report['model_tree_gib']=round(sum(p.stat().st_size for p in model.rglob('*') if p.is_file())/2**30,3) if model.exists() else 0
for name in ('setup','download','torch-setup'):
    path=root/'jev-service'/f'{name}.log'
    if path.exists():report[name+'_tail']=path.read_text(errors='replace').splitlines()[-4:]
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3) as r:report['http_health']=r.status
except Exception as e:report['http_health']=type(e).__name__
probe=root/'jev-native-v4/probe.json'
if probe.exists():report['probe']=json.loads(probe.read_text())
progress=root/'jev-native-v4/progress.json'
if progress.exists():report['scoring']=json.loads(progress.read_text())
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
Path('/root/jev-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
