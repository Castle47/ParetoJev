import fcntl
import importlib.util
from pathlib import Path
import sys

import pytest

from paretojev.utils.io import atomic_write_json


def loader():
    path=Path(__file__).parents[1]/'scripts/prepare_kuairand_27k.py'
    spec=importlib.util.spec_from_file_location('kuairand_cli_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_verified_core_extraction_can_run_while_other_asset_downloads(tmp_path,monkeypatch):
    module=loader();called=[]
    monkeypatch.setattr(sys,'argv',['prepare','extract','--root',str(tmp_path)])
    monkeypatch.setattr(module,'extract_27k',lambda *args,**kwargs:called.append(args))
    with (tmp_path/'acquisition.lock').open('a') as download_lock:
        fcntl.flock(download_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        module.main()
    assert len(called)==1


def test_full_audit_refuses_in_progress_extraction(tmp_path,monkeypatch):
    module=loader();d=tmp_path/'extracted';d.mkdir()
    atomic_write_json(d/'extraction_manifest.json',{'status':'in_progress'})
    monkeypatch.setattr(sys,'argv',['prepare','audit','--root',str(tmp_path)])
    def forbidden(*args,**kwargs):raise AssertionError('Partial extraction was audited')
    monkeypatch.setattr(module,'audit_directory',forbidden)
    with pytest.raises(ValueError,match='complete atomic extraction'):
        module.main()
