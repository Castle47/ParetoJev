import json
import numpy as np
import pandas as pd
import pytest
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_features_v2 import META, CORE_FIELDS, feature_rows, make_metadata
from paretojev.utils.provenance import sha256_file


def sample():
    e=np.zeros(4,dtype=EVENT);e['time']=[10,20,20,30];e['item']=[0,1,0,2]
    e['duration']=[1000,2000,3000,0];e['tab']=[0,1,0,1]
    m=np.zeros(3,dtype=META);m['author']=[5,5,7];m['upload_day']=0
    m['duration']=1000;m['width']=720;m['height']=1280
    return e,m


def test_ties_future_and_feedback_cannot_enter_features():
    e,m=sample();x,d=feature_rows(e,np.array([1,2]),m)
    assert (x[:,CORE_FIELDS.index('history_length_fraction')]==1/50).all()
    assert x[0,CORE_FIELDS.index('candidate_repeat_fraction')]==0
    assert x[1,CORE_FIELDS.index('candidate_repeat_fraction')]==1
    changed=e.copy();changed['labels']=127;changed[-1]['duration']=999999
    assert np.array_equal(x,feature_rows(changed,np.array([1,2]),m)[0])
    assert d['rows']==2


def test_future_upload_and_missing_dates_mask_content():
    e,m=sample();m['upload_day'][1]=99;m['upload_day'][0]=-1
    a,_=feature_rows(e,np.array([1]),m)
    m['author'][:2]=99;m['duration'][:2]=900000;m['width'][:2]=3000
    b,_=feature_rows(e,np.array([1]),m)
    np.testing.assert_array_equal(a,b)
    assert np.isnan(a[0,len(CORE_FIELDS)])
    assert a[0,len(CORE_FIELDS)+5]==0
    assert a[0,-1]==0


def test_empty_history_is_finite_and_zero():
    e,m=sample();x,_=feature_rows(e,np.array([0]),m)
    assert np.isfinite(x).all()
    assert x[0,11]==0 and x[0,-1]==0 and x[0,-2]==0
    with pytest.raises(ValueError,match='Chronological'):
        feature_rows(e[::-1],np.array([0]),m)


def test_metadata_full_identity_hash_and_cache(tmp_path):
    source=tmp_path/'basic.csv'
    pd.DataFrame(dict(video_id=[0,1],author_id=[1,2],upload_dt=['2022-04-08','bad'],
       video_duration=[1000,0],server_width=[720,0],server_height=[1280,-1])).to_csv(source,index=False)
    extraction=tmp_path/'extraction.json';extraction.write_text(json.dumps({'files':{
       'KuaiRand-27K/data/video_features_basic_27k.csv':{'sha256':sha256_file(source)}}}))
    out=tmp_path/'out';out.mkdir()
    m,r=make_metadata(source,extraction,tmp_path/'cache',out,item_count=2,chunksize=1)
    assert r['rows']==2 and r['missing']['upload_day']==1
    assert m['author'].tolist()==[1,2]
    assert np.isnan(m['duration'][1])
    again,_=make_metadata(source,extraction,tmp_path/'cache',out,item_count=2)
    assert again.tobytes() == m.tobytes()
