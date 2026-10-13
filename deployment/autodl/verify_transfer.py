"""Reassemble only the registered request packet and check both byte hashes."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

root=Path('/root')
manifest=json.loads((root/'transfer.json').read_text())
expected='21e1d02f77f4cb6d9db20e93b02893ea04ca40356e847644b91c82268afec636'
assert manifest['compressed_sha256']==expected
tmp=root/'jev_input.jsonl.gz.assembling'
with tmp.open('wb') as out:
    for i in range(13):
        with (root/f'jev_input.part-{i:03d}').open('rb') as part:shutil.copyfileobj(part,out)
with tmp.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected, 'Compressed bytes changed'
with gzip.open(tmp,'rb') as f:raw=hashlib.file_digest(f,'sha256').hexdigest()
assert raw==manifest['source_input_sha256'], 'Uncompressed bytes changed'
tmp.replace(root/'jev_input.jsonl.gz')
result=dict(verified=True,compressed_sha256=actual,source_input_sha256=raw,slates=36000,goals=7,
            independent_test_included=False,labels_included=False)
(root/'transfer-verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
