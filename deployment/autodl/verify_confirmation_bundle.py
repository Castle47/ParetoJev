"""CPU-only upload/model preflight; no API calls or GPU model loading."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(a):
    manifest=json.loads((a.input_dir/'transfer.json').read_text())
    dev=a.input_dir/'development-registration.json'
    if sha(dev)!=manifest['development_registration_sha256'] or manifest['labels_included'] or not manifest['independent_test_included']:
        raise ValueError('Upload scope or development identity differs')
    if manifest['slates']!=50000 or manifest['planned_decisions']!=350000:
        raise ValueError('Expected frozen50000-slate packet')
    if sha(a.input_dir/'confirmation_config.json')!=manifest['confirmation_config_sha256']:
        raise ValueError('Frozen confirmation protocol differs')
    payload=a.input_dir/'jev_input.jsonl.gz'
    if sha(payload)!=manifest['compressed_sha256']:raise ValueError('Uploaded input hash differs')
    with gzip.open(payload,'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=manifest['source_input_sha256']:
            raise ValueError('Decoded input hash differs')
    source=json.loads(dev.read_text())['source'];verified=0
    for name,expected in source['files'].items():
        path=Path('/root/torch_head_server.py') if name=='custom/torch_head_server.py' else a.model_dir/name
        if sha(path)!=expected:raise ValueError('Producer file differs:'+name)
        verified+=1
    for name,expected in source['packages'].items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=expected:raise ValueError('Producer package differs:'+name)
    result=dict(input_and_model_hashes_verified=True,producer_files=verified,
        compressed_sha256=manifest['compressed_sha256'],development_registration_sha256=sha(dev),
        packages_match=True,gpu_inference_started=False)
    (a.input_dir/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=Path('/root/autodl-tmp/JEV-27B'))
    main(p.parse_args())
