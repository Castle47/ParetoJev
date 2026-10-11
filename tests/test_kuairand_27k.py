import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pandas as pd
import pytest

from paretojev.data.acquisition import download_asset, extract_27k, safe_member_path
from paretojev.data.kuairand_27k import (
    ASSETS, CORE_FILES, audit_log, discover_data, feature_contract, storage_plan,
)
from paretojev.utils.io import atomic_write_json


def row(user, item, timestamp, click=0):
    return {'user_id':user,'video_id':item,'date':20220408,'time_ms':timestamp,'tab':1,'is_rand':0,
        'play_time_ms':20000,'duration_ms':30000,'is_click':click,'long_view':1,'is_like':0,
        'is_follow':0,'is_comment':0,'is_forward':0,'is_hate':0}


def test_streaming_audit_matches_hash_and_is_chunk_invariant(tmp_path):
    p=tmp_path/'log.csv'
    pd.DataFrame([row(0,1,300,1),row(1,2,100),row(0,3,200,1)]).to_csv(p,index=False)
    a=audit_log(p,random_log=False,chunksize=1,user_count=3,item_count=5)
    b=audit_log(p,random_log=False,chunksize=100,user_count=3,item_count=5)
    assert a==b
    assert a['source_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert a['positive_counts']['is_click']==2 and a['global_timestamp_backsteps']==1
    assert a['user_event_counts']==[2,1,0]
    prefix=audit_log(p,random_log=False,chunksize=1,max_rows=2,user_count=3,item_count=5)
    assert prefix['scope']=='prefix_only' and prefix['source_sha256'] is None and prefix['rows']==2


@pytest.mark.parametrize('column,value,message',[
    ('is_like',2,'Nonbinary'),('is_rand',1,'tracks mixed'),('user_id',4,'ID outside'),('tab',99,'Unknown tab')])
def test_invalid_labels_ids_and_track_are_not_silently_repaired(tmp_path,column,value,message):
    r=row(0,1,100);r[column]=value;p=tmp_path/'bad.csv';pd.DataFrame([r]).to_csv(p,index=False)
    with pytest.raises(ValueError,match=message):audit_log(p,random_log=False,user_count=3,item_count=5)


def test_requires_all_27k_shards_and_rejects_pure_fallback(tmp_path):
    (tmp_path/'log_standard_4_08_to_4_21_pure.csv').touch()
    with pytest.raises(ValueError,match='ten core'):discover_data(tmp_path)
    d=tmp_path/'KuaiRand-27K/data';d.mkdir(parents=True)
    for name in CORE_FILES:(d/name).touch()
    assert discover_data(tmp_path)==d
    for name in CORE_FILES:(tmp_path/name).touch()
    with pytest.raises(ValueError,match='exactly one'):discover_data(tmp_path)


def test_resource_plan_blocks_small_disk_but_credits_existing_acquisition(tmp_path,monkeypatch):
    monkeypatch.setattr('paretojev.data.kuairand_27k.shutil.disk_usage',lambda _:SimpleNamespace(free=4*2**30))
    p=storage_plan(tmp_path)
    assert p['status']=='blocked_storage'
    d=tmp_path/'downloads';d.mkdir();(d/ASSETS[0]['name']).write_bytes(b'abc')
    p=storage_plan(tmp_path)
    assert p['existing_dataset_bytes']==3
    assert p['required_remaining_free_gib'] < p['initial_workspace_budget_gib']
    f=feature_contract()
    assert len(f['user_fields_expected'])==30 and len(f['item_basic_fields_expected'])==11
    assert f['approved_real_input_feature_count'] is None and f['training_ready'] is False


class Response:
    def __init__(self,body,status=200,headers=None):
        self.body=body;self.status_code=status;self.headers=headers or {'Content-Length':str(len(body))}
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def raise_for_status(self):pass
    def iter_content(self,_):yield self.body


class Session:
    def __init__(self,response):self.response=response;self.calls=[]
    def get(self,*args,**kwargs):self.calls.append((args,kwargs));return self.response


def asset(data):
    return {'name':'data.bin','url':'https://example.test/data.bin','md5':hashlib.md5(data).hexdigest()}


def test_download_verified_and_wrong_checksum_not_promoted(tmp_path):
    data=b'actual bytes';spec=asset(data)
    s=Session(Response(data));r=download_asset(spec,tmp_path,session=s,reserve_bytes=0)
    assert r['sha256']==hashlib.sha256(data).hexdigest() and r['status']=='downloaded_verified'
    assert download_asset(spec,tmp_path,session=s,reserve_bytes=0)['status']=='verified_existing'
    assert len(s.calls)==1
    other=tmp_path/'other'
    with pytest.raises(ValueError,match='checksum mismatch'):
        download_asset(spec,other,session=Session(Response(b'wrong')),reserve_bytes=0)
    assert not (other/'data.bin').exists() and (other/'data.bin.part').exists()


def test_resume_requires_exact_range_and_preserves_partial(tmp_path):
    full=b'abcdef';spec=asset(full)
    partial=tmp_path/'data.bin.part';partial.write_bytes(b'abc')
    atomic_write_json(tmp_path/'data.bin.source.json',spec)
    with pytest.raises(ValueError,match='resume range'):
        download_asset(spec,tmp_path,session=Session(Response(full)),reserve_bytes=0)
    assert partial.read_bytes()==b'abc'
    s=Session(Response(b'def',206,{'Content-Range':'bytes 3-5/6'}))
    download_asset(spec,tmp_path,session=s,reserve_bytes=0)
    assert s.calls[0][1]['headers']['Range']=='bytes=3-'
    assert (tmp_path/'data.bin').read_bytes()==full


def test_download_low_disk_fails_before_network(tmp_path,monkeypatch):
    monkeypatch.setattr('paretojev.data.acquisition.shutil.disk_usage',lambda _:SimpleNamespace(free=0))
    s=Session(Response(b'x'))
    with pytest.raises(OSError,match='Insufficient storage'):download_asset(asset(b'x'),tmp_path,session=s)
    assert not s.calls and not (tmp_path/'data.bin.part').exists()


@pytest.mark.parametrize('name',['../evil','/tmp/evil','KuaiRand-27K/../../evil','KuaiRand-Pure/a','KuaiRand-27K\\evil'])
def test_unsafe_or_wrong_variant_archive_paths_rejected(name):
    with pytest.raises(ValueError):safe_member_path(name)


def make_archive(path,files):
    with tarfile.open(path,'w:gz') as archive:
        for name,body in files.items():
            member=tarfile.TarInfo(name);member.size=len(body);archive.addfile(member,io.BytesIO(body))
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_extraction_replay_and_existing_file_tamper_guard(tmp_path):
    p=tmp_path/'data.tar.gz';md5=make_archive(p,{'KuaiRand-27K/data/a.csv':b'a,b\n1,2\n'})
    out=tmp_path/'out'
    a=extract_27k(p,out,expected_md5=md5,reserve_bytes=0)
    assert a['status']=='complete'
    assert extract_27k(p,out,expected_md5=md5,reserve_bytes=0)==a
    (out/'KuaiRand-27K/data/a.csv').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Existing extracted file differs'):
        extract_27k(p,out,expected_md5=md5,reserve_bytes=0)


def test_extraction_temporary_symlink_does_not_overwrite_external_file(tmp_path):
    p=tmp_path/'data.tar.gz';md5=make_archive(p,{'KuaiRand-27K/data/a.csv':b'a\n1\n'})
    out=tmp_path/'out';d=out/'KuaiRand-27K/data';d.mkdir(parents=True)
    other=tmp_path/'keep.txt';other.write_text('keep')
    (d/'a.csv.extracting').symlink_to(other)
    with pytest.raises(ValueError,match='symlinks'):
        extract_27k(p,out,expected_md5=md5,reserve_bytes=0)
    assert other.read_text()=='keep'


class RangeSession:
    def __init__(self, data, interrupt=False):
        self.data=data;self.calls=[];self.interrupt=interrupt
    def get(self,url,**kwargs):
        import requests
        requested=kwargs['headers']['Range'];self.calls.append(requested)
        start,end=map(int,requested.removeprefix('bytes=').split('-'))
        end=min(end,len(self.data)-1)
        response=Response(self.data[start:end+1],206,{'Content-Range':f'bytes {start}-{end}/{len(self.data)}'})
        if self.interrupt:
            self.interrupt=False
            def chunks(_):
                yield self.data[start:start+2]
                raise requests.ConnectionError('read timed out')
            response.iter_content=chunks
        return response


def test_bounded_ranges_and_retry_preserve_every_byte(tmp_path,monkeypatch):
    monkeypatch.setattr('paretojev.data.acquisition.time.sleep',lambda _:None)
    data=b'abcdef';s=RangeSession(data,interrupt=True)
    result=download_asset(asset(data),tmp_path,session=s,reserve_bytes=0,request_bytes=3,max_transport_retries=1)
    assert s.calls==['bytes=0-2','bytes=2-4','bytes=5-7']
    assert (tmp_path/'data.bin').read_bytes()==data
    assert result['md5']==hashlib.md5(data).hexdigest()


def test_bounded_mode_rejects_range_ignored_by_server(tmp_path):
    with pytest.raises(ValueError,match='exact HTTP range'):
        download_asset(asset(b'abcdef'),tmp_path,session=Session(Response(b'abcdef')),reserve_bytes=0,request_bytes=3)
    assert not (tmp_path/'data.bin.part').exists()


def test_bounded_retry_budget_stops_after_retaining_valid_prefix(tmp_path):
    import requests
    s=RangeSession(b'abcdef',interrupt=True)
    with pytest.raises(requests.ConnectionError):
        download_asset(asset(b'abcdef'),tmp_path,session=s,reserve_bytes=0,request_bytes=3,max_transport_retries=0)
    assert len(s.calls)==1 and (tmp_path/'data.bin.part').read_bytes()==b'ab'
