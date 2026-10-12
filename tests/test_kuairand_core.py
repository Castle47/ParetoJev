import numpy as np
from paretojev.data.kuairand_pilot import EVENT
from paretojev.data.kuairand_features_v2 import META
from paretojev.data.kuairand_core import eligible_blocks,strata_for,common_time_features


def test_fixed_blocks_and_rare_priority():
    e=np.zeros(12,dtype=EVENT);e['time']=np.arange(12)*10;e['item']=np.arange(12)
    e['tab'][5]=1;e['labels'][1]=8;e['labels'][2]=64
    blocks=eligible_blocks(e,0,120,3,50)
    assert blocks.tolist()==[[0,1,2],[6,7,8],[9,10,11]]
    assert strata_for(e,blocks,[64,8,32,16,4]).tolist()==[0,5,5]
    e['item'][10]=e['item'][9]
    assert len(eligible_blocks(e,0,120,3,50))==2


def test_common_boundary_excludes_earlier_candidates_in_same_slate():
    e=np.zeros(5,dtype=EVENT);e['time']=[10,20,20,30,40];e['item']=np.arange(5);e['duration']=1000
    meta=np.zeros(5,dtype=META);meta['upload_day']=0;meta['duration']=1000;meta['width']=100;meta['height']=200
    x,history,t0,_=common_time_features(e,e[[4,2,3]],meta,history_length=3)
    assert t0==20 and history['time'].tolist()==[10]
    assert np.all(x[:,11]==np.float32(1/3))
    altered=e.copy();altered['labels']=127
    np.testing.assert_array_equal(x,common_time_features(altered,altered[[4,2,3]],meta,history_length=3)[0])
    # A later exposure cannot change any candidate's features at the shared time.
    altered['item'][3]=0
    np.testing.assert_array_equal(x,common_time_features(altered,e[[4,2,3]],meta,history_length=3)[0])


def test_choice_validation_rejects_unpinned_and_permuted_results():
    import pytest
    from paretojev.jev.public_choice import validate_choice
    from paretojev.jev.client import DecisionAPIError
    r=dict(model='m',revision='r',kind='choice',options=['A','B'],probabilities=[.2,.8],
        choice_index=1,choice='B',image_used=False,elapsed_seconds=.1)
    assert validate_choice(r,model='m',revision='r',options=['A','B'])==[.2,.8]
    for change in ({'revision':'other'},{'options':['B','A']},{'probabilities':[float('nan'),1.]},
        {'choice':'A'},{'probabilities':[.3,.8]},{'image_used':True}):
        with pytest.raises(DecisionAPIError):validate_choice({**r,**change},model='m',revision='r',options=['A','B'])
