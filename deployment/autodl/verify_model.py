"""Verify pinned publisher blob hashes; optionally repair only absent/bad files."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import time


def check(path,spec):
    if not path.is_file():return 'missing'
    if path.stat().st_size!=spec['size']:return 'wrong_size'
    if spec.get('lfs'):
        with path.open('rb') as f:value=hashlib.file_digest(f,'sha256').hexdigest()
        return 'ok' if value==spec['lfs']['sha256'] else 'wrong_sha256'
    h=hashlib.sha1();h.update(f'blob {path.stat().st_size}\0'.encode());h.update(path.read_bytes())
    return 'ok' if h.hexdigest()==spec['blobId'] else 'wrong_git_blob'


def main(a):
    manifest=json.loads(a.manifest.read_text());root=a.model_dir.resolve();results={}
    for spec in manifest['files']:
        rel=PurePosixPath(spec['rfilename'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe publisher path')
        path=root/rel
        if not path.resolve().is_relative_to(root):raise ValueError('Unexpected model symlink')
        status=check(path,spec)
        if status!='ok' and a.repair:
            if shutil.disk_usage(root).free<spec['size']*1.1:raise RuntimeError('Insufficient repair headroom')
            if path.exists():path.rename(path.with_name(path.name+'.quarantine-'+str(time.time_ns())))
            from huggingface_hub import hf_hub_download
            print('Repairing',str(rel),status,flush=True)
            hf_hub_download(manifest['repo'],str(rel),revision=manifest['revision'],local_dir=str(root),force_download=True)
            status=check(path,spec)
        results[str(rel)]=status
        print(str(rel),status,flush=True)
    result=dict(revision=manifest['revision'],all_publisher_hashes_match=all(v=='ok' for v in results.values()),files=results,
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_publisher_hashes_match']:raise RuntimeError('Model integrity check failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('/root/model_manifest.json'))
    p.add_argument('--model-dir',type=Path,default=Path('/root/autodl-tmp/JEV-27B'))
    p.add_argument('--output',type=Path,default=Path('/root/autodl-tmp/jev-service/model-verified.json'))
    p.add_argument('--repair',action='store_true');main(p.parse_args())
