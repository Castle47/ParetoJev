import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_confirmation_sampling import audit_sampling
from audit_multiobjective_cohort import partition_by_user
from prepare_emer_jev import DAY_EVENT
from paretojev.utils.io import atomic_write_json


def test_sampling_replay_recovers_known_uniform_priority_sample(tmp_path):
    c=dict(fresh_test_start_ms=0,fresh_test_end_ms=100000,slate_size=10,
        maximum_slate_span_ms=300000,stratum_priority_bits=[64,8,32,16,4],sample_seed=77,
        test_stratum_budgets=[1,0,0,0,1,0])
    events=np.zeros(50,dtype=DAY_EVENT)
    events['record']=np.arange(50);events['item']=np.arange(50);events['time']=(np.arange(50)+1)*1000
    events['user'][30:]=512;events['labels'][:30]=64;events['labels'][30:]=4
    partition_by_user(events,tmp_path/'audit-partitions-v1','source-sha')
    atomic_write_json(tmp_path/'preparation.json',dict(sources={'sha256':'source-sha'},populations=[3,0,0,0,2,0]))
    chosen=[]
    for i,(user,n,start) in enumerate([(0,3,0),(512,2,30)]):
        block=int(np.argmax(np.random.default_rng(np.random.SeedSequence([77,user])).random(n)))
        ids=np.arange(start+10*block,start+10*(block+1))
        chosen.append(ids[np.random.default_rng(np.random.SeedSequence([77,user,i])).permutation(10)])
    np.savez(tmp_path/'raw_slates.npz',records=np.array(chosen),user=np.array([0,512]),stratum=np.array([0,4]),weight=np.array([3.,2.]))
    result=audit_sampling(tmp_path,c)
    assert result['exact_selection_replayed'] and result['populations']==[3,0,0,0,2,0]
    np.savez(tmp_path/'raw_slates.npz',records=np.array(chosen)[:,::-1],user=np.array([0,512]),stratum=np.array([0,4]),weight=np.array([3.,2.]))
    with pytest.raises(ValueError,match='identity differs'):audit_sampling(tmp_path,c)
