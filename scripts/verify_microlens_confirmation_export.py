"""Restore source and exported data in a temporary directory, without HTTP."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import run_microlens_confirmatory as run
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import sha256_file


if __name__ == '__main__':
    archive = Path('../ParetoJev-confirmatory-replay-2026-10-09.tar.gz').resolve()
    exported = read_json(archive.with_name('ParetoJev-confirmatory-replay-2026-10-09.manifest.json'))
    if sha256_file(archive) != exported['archive_sha256']:
        raise ValueError('Exported archive digest differs')
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source_tar = subprocess.check_output(['git','archive','--format=tar',source_commit])
    with tempfile.TemporaryDirectory(prefix='confirm-offline-restore-',dir='../../work') as directory:
        target = Path(directory).resolve()
        with tarfile.open(fileobj=io.BytesIO(source_tar)) as source:
            source.extractall(target,filter='data')
        with tarfile.open(archive,'r:gz') as data:
            data.extractall(target,filter='data')
        for entry in exported['files']:
            path = target / entry['path']
            path.resolve().relative_to(target)
            if sha256_file(path) != entry['sha256']:
                raise ValueError(f'Restored file differs: {entry["path"]}')
        program = '''
import sys,contextlib,io
from pathlib import Path
from unittest.mock import patch
import paretojev
assert Path(paretojev.__file__).resolve().is_relative_to(Path.cwd())
sys.path.insert(0,str(Path('scripts').resolve()))
import run_microlens_confirmatory as core
import resume_microlens_confirmatory_transport as transport
from paretojev.utils.provenance import sha256_file
paths=[core.ROOT/name for name in ['scores.jsonl','requests.sqlite3','http_attempts.jsonl']]
before=[sha256_file(p) for p in paths]
with patch('requests.sessions.Session.request',side_effect=AssertionError('HTTP prohibited in restored replay')):
    with contextlib.redirect_stdout(io.StringIO()):transport.main()
    core.replay()
assert before==[sha256_file(p) for p in paths]
print('Fresh-directory source, cached collection, checkpoint and metrics verified')
'''
        environment = dict(os.environ); environment['PYTHONPATH'] = str(target/'src')
        completed = subprocess.run([sys.executable,'-c',program],cwd=target,env=environment,
                                   text=True,capture_output=True,check=True)
        print(completed.stdout,flush=True)
    result = {'status':'passed','source_commit_checked':source_commit,
        'archive_sha256':exported['archive_sha256'],'data_files_checked':len(exported['files']),
        'restored_source_imported':True,'all_restored_data_hashes_match':True,
        'http_blocked_for_entire_replay':True,'collection_checkpoint_and_evaluation_replay_exact':True,
        'scope':'same-machine fresh-directory replay; not an independent training or cross-hardware run'}
    atomic_write_json(run.REPORT/'portable_replay.json',result)
    print(json.dumps(result,indent=2),flush=True)
