import importlib.util
from pathlib import Path
import time

import pytest

spec=importlib.util.spec_from_file_location('native_jev',Path(__file__).resolve().parents[1]/'deployment/autodl/score_native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)


def response():
    return dict(model=native.MODEL,protocol=native.PROTOCOL,kind='choice',adaptation='native',
        options=['A','B'],probabilities=[.2,.8],choice_index=1,choice='B',num_model_requests=1,elapsed_seconds=.1)


def test_native_response_rejects_wrong_identity_and_invalid_probability():
    assert native.validate(response(),['A','B'])==[.2,.8]
    for change in [dict(protocol='wrong'),dict(probabilities=[float('nan'),.8]),
        dict(probabilities=[.2,.7]),dict(choice_index=0),dict(options=['B','A']),dict(num_model_requests=2)]:
        with pytest.raises(ValueError):native.validate(dict(response(),**change),['A','B'])


def test_resumed_ledger_keeps_attempt_budget_and_payload_identity(tmp_path):
    p=tmp_path/'native.sqlite3';one=native.Ledger(p,1,time.time()+100)
    one.reserve(0,0);one.put(0,0,'original',response(),.1);one.db.close()
    two=native.Ledger(p,1,time.time()+100)
    assert two.cached(0,0,'original')==response()
    with pytest.raises(ValueError,match='input changed'):two.cached(0,0,'changed')
    with pytest.raises(RuntimeError,match='budget'):two.reserve(1,0)
    two.db.close()


def test_expired_deadline_blocks_before_any_request_reservation(tmp_path):
    ledger=native.Ledger(tmp_path/'native.sqlite3',100,time.time()-1)
    with pytest.raises(TimeoutError):ledger.reserve(0,0)
    assert ledger.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0
    ledger.db.close()


def test_torch_head_prompt_matches_published_bare_template():
    path=Path(__file__).resolve().parents[1]/'deployment/autodl/torch_head_server.py'
    s=importlib.util.spec_from_file_location('torch_head_test',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    assert m.prompt(dict(kind='choice',state='context',question='which?',options=['A','B']))==(
        '[kind] choice\n[state] context\n[question] which?\n[options]\nA) A\nB) B\n[decision]:')
    with pytest.raises(ValueError):m.prompt(dict(kind='choice',state='x',question='q',options=['A','B'],thinking='on'))


def test_publisher_hash_checks_reject_same_size_corruption(tmp_path):
    import hashlib
    path=Path(__file__).resolve().parents[1]/'deployment/autodl/verify_model.py'
    s=importlib.util.spec_from_file_location('verify_model_test',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    p=tmp_path/'weights';p.write_bytes(b'abcd')
    spec=dict(size=4,lfs=dict(sha256=hashlib.sha256(b'abcd').hexdigest()))
    assert m.check(p,spec)=='ok'
    p.write_bytes(b'abce');assert m.check(p,spec)=='wrong_sha256'
    p.write_bytes(b'abc');assert m.check(p,spec)=='wrong_size'
    q=tmp_path/'config';q.write_bytes(b'{}')
    assert m.check(q,dict(size=2,blobId=hashlib.sha1(b'blob 2\0{}').hexdigest()))=='ok'


def test_grouped_wire_response_is_preserved_atomically(tmp_path):
    import hashlib,json,sqlite3
    row=dict(slate_id=0,state='context',questions=[f'q{i}' for i in range(7)],options=['A','B'])
    children=[]
    for i in range(7):
        r=response();r.pop('num_model_requests');r.update(implementation='pytorch-peft-group-v3',question_index=i,group_shared_forwards=2);children.append(r)
    group=dict(model=native.MODEL,protocol=native.PROTOCOL,implementation='pytorch-peft-group-v3',
        model_forward_calls=2,question_count=7,results=children)
    assert len(native.validate_group(group,row))==7
    ledger=native.Ledger(tmp_path/'native.sqlite3',10,time.time()+100)
    ledger.put_group(row,group,.2)
    assert json.loads(ledger.db.execute('SELECT raw_json FROM groups').fetchone()[0])==group
    for j in range(7):
        key=hashlib.sha256(native.canonical(native.payload(row,j)).encode()).hexdigest()
        assert ledger.cached(0,j,key)==children[j]
    with pytest.raises(sqlite3.IntegrityError):ledger.put_group(row,group,.2)
    assert ledger.db.execute('SELECT COUNT(*) FROM results').fetchone()[0]==7
    ledger.db.close()
