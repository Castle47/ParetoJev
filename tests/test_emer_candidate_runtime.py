import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deployment/emer_candidate'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from candidate_runtime import user_routes,validate_inputs
from run_emer_user_cv import user_bucket


def valid_inputs():
    n=4
    return dict(raw_features=np.zeros((n,10,26),np.float32),behavior_scores=np.full((n,10,7),.5,np.float32),
        jev_scores=np.full((n,10,7),.1,np.float32),jev_entropy=np.ones((n,10,7),np.float32),
        jev_confidence=np.zeros((n,10,7),np.float32),user_ids=np.array([1,2,1,3]),
        candidate_ids=np.tile(np.arange(10),(n,1)),mask=np.ones((n,10),bool),popularity=np.ones((n,10,1)),history_length=np.ones((n,1)))


def test_routing_is_original_hash_and_order_invariant():
    ids=np.arange(1000,dtype=np.int64);expected=np.array([user_bucket(x,'outer',2026101407,3) for x in ids])
    np.testing.assert_array_equal(user_routes(ids),expected)
    p=np.random.default_rng(7).permutation(len(ids));np.testing.assert_array_equal(user_routes(ids[p]),expected[p])
    with pytest.raises(ValueError):user_routes(np.array([1.2]))
    with pytest.raises(ValueError):user_routes(np.array([-1]))


def test_schema_rejects_labels_and_wrong_encoder_input():
    x=valid_inputs();x['raw_features'][0,0,2]=np.nan
    out=validate_inputs(x);assert out['popularity'].shape==(4,10) and out['history_length'].shape==(4,)
    with pytest.raises(ValueError):validate_inputs(dict(x,labels=np.zeros((4,10,7))))
    with pytest.raises(ValueError):validate_inputs(dict(x,raw_features=np.zeros((4,10,65))))


def test_schema_rejects_wrong_choices_and_metadata():
    x=valid_inputs();x['jev_scores']*=2
    with pytest.raises(ValueError):validate_inputs(x)
    x=valid_inputs();x['jev_entropy'][0,0,0]=.8
    with pytest.raises(ValueError):validate_inputs(x)
    x=valid_inputs();x['candidate_ids'][0,1]=0
    with pytest.raises(ValueError):validate_inputs(x)
    x=valid_inputs();x['raw_features'][0,0,0]=1.5
    with pytest.raises(ValueError):validate_inputs(x)
