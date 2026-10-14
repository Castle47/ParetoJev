import json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_emer_routed_confirmation as runner


def test_missing_or_partial_scores_block_before_any_model_or_materialization(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Reached model preparation before real scores')
    monkeypatch.setattr(runner,'verify_preparation',forbidden)
    with pytest.raises(ValueError,match='Real score directory required'):
        runner.evaluate(SimpleNamespace(scores=None),{})
    scores=tmp_path/'scores';scores.mkdir()
    for name in runner.SCORE_FILES:(scores/name).write_text('{}')
    (scores/'complete.json').write_text(json.dumps(dict(complete=False,valid_decisions=349999,http_groups=49999)))
    with pytest.raises(ValueError,match='All350000'):
        runner.evaluate(SimpleNamespace(scores=scores),{})


def test_mixed_service_cache_cannot_pass_initial_readiness(tmp_path):
    for name in runner.SCORE_FILES:(tmp_path/name).write_text('{}')
    (tmp_path/'complete.json').write_text(json.dumps(dict(complete=True,public_cache_mixed=True,valid_decisions=350000,http_groups=50000)))
    with pytest.raises(ValueError,match='All350000'):runner.require_scores(tmp_path)


def test_bundle_override_cannot_bypass_frozen_candidate(tmp_path,monkeypatch):
    bundle=tmp_path/'other-bundle';bundle.mkdir();(bundle/'manifest.json').write_text('{}')
    c=dict(score_protocol={'path':'unused.json'},candidate_manifest_sha256='0'*64)
    (tmp_path/'registration.json').write_text(json.dumps(dict(config=c,data_root=str(tmp_path.resolve()),inputs={})))
    def forbidden(*args):raise AssertionError('Loaded unregistered model')
    monkeypatch.setattr(runner,'Candidate',forbidden)
    with pytest.raises(ValueError,match='Requested candidate differs'):
        runner.verify_preparation(SimpleNamespace(root=tmp_path,output=tmp_path,bundle=bundle),c)


def test_data_root_override_requires_new_preparation(tmp_path):
    c={};(tmp_path/'registration.json').write_text(json.dumps(dict(config=c,data_root='/different-root',inputs={})))
    with pytest.raises(ValueError,match='Registered data root changed'):
        runner.verify_preparation(SimpleNamespace(root=tmp_path,output=tmp_path),c)
