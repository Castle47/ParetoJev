import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_emer_scale import paired_precision,projected_slates


def test_cluster_precision_matches_mean_se_and_global_weight_invariance():
    b=np.zeros((4,1));f=np.array([[.1],[.2],[-.1],[.4]]);d=np.ones_like(b)
    base,delta,se=paired_precision(b,f,d,np.ones(4),np.arange(4))
    np.testing.assert_allclose(se,f.std(ddof=1)/2)
    np.testing.assert_allclose(delta,f.mean())
    for x,y in zip((base,delta,se),paired_precision(b,f,d,np.full(4,19.),np.arange(4))):
        np.testing.assert_allclose(x,y)


def test_duplicate_rows_within_user_do_not_create_independent_information():
    b=np.zeros((3,1));f=np.array([[.1],[.4],[-.2]]);d=np.ones_like(b)
    original=paired_precision(b,f,d,np.ones(3),np.arange(3))
    repeated=paired_precision(b.repeat(5,0),f.repeat(5,0),d.repeat(5,0),np.ones(15),np.arange(3).repeat(5))
    for x,y in zip(original,repeated):np.testing.assert_allclose(x,y)


def test_support_and_no_variance_do_not_yield_fake_power():
    with pytest.raises(ValueError,match='support'):
        paired_precision(np.zeros((2,1)),np.zeros((2,1)),np.zeros((2,1)),np.ones(2),[0,1])
    assert projected_slates(600,0,.01) is None
    assert abs(projected_slates(600,.01,.01)/projected_slates(600,.01,.02)-4)<.01


def test_two_split_sampler_and_exact_inclusion_weights(tmp_path,monkeypatch):
    import prepare_kuairand_core as prep
    from paretojev.data.kuairand_pilot import EVENT
    from paretojev.utils.io import atomic_write_json,read_json
    from paretojev.utils.provenance import sha256_file
    root=tmp_path/'root';store=root/'behavior-pilot-v1';store.mkdir(parents=True)
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(prep,'ITEM_COUNT',20)
    # Two disjoint windows, each containing two eligible blocks in every stratum.
    e=np.zeros(48,dtype=EVENT);e['time']=np.arange(48)+1;e['item']=np.arange(48)%20
    for s in range(2):
        for h,bit in enumerate([64,8,32,16,4,0]):e['labels'][s*24+h*4:s*24+h*4+4]=bit
    e.tofile(store/'events.bin');np.save(store/'offsets.npy',np.array([0,len(e)]))
    atomic_write_json(store/'events.json',dict(offsets_sha256=sha256_file(store/'offsets.npy'),store_sha256=sha256_file(store/'events.bin')))
    c=dict(split_names=['train','validation'],boundaries_ms=[1,25,49],sample_budgets={'train':[1]*6,'validation':[1]*6},
        slate_size=2,maximum_slate_span_ms=10,stratum_priority_bits=[64,8,32,16,4],seed=7)
    prep.select(root,out,c);z=np.load(out/'selection.npz')
    assert len(z['user'])==12 and set(z['split'])=={0,1}
    np.testing.assert_array_equal(z['weight'],np.full(12,2))
    assert len(np.unique(z['records']))==24
    assert read_json(out/'selection.json')['population_by_split_stratum']==[[2]*6,[2]*6]


def test_scaled_driver_requires_native_scores_and_preserves_common_inputs(tmp_path):
    from run_emer_scale import development_data
    from paretojev.utils.io import atomic_write_json
    from paretojev.utils.provenance import canonical_hash,sha256_file
    data=tmp_path/'data';out=tmp_path/'out';out.mkdir();(data/'jev').mkdir(parents=True)
    c=dict(split_names=['train','validation'],objectives=['a','b'],jev=dict(model='test',revision='fixed'))
    for i,name in enumerate(c['split_names']):
        np.savez_compressed(out/f'{name}.npz',candidate_features=np.ones((1,2,3)),behavior_scores=np.full((1,2,2),.4),
            jev_scores=np.zeros((1,2,2)),jev_entropy=np.zeros((1,2,2)),jev_confidence=np.zeros((1,2,2)),
            labels=np.ones((1,2,2)),mask=np.ones((1,2),bool),popularity=np.ones((1,2,1)),history_length=np.ones((1,1)),
            slate_weights=np.ones(1),user_ids=np.array([i]),candidate_ids=np.array([[i*2,i*2+1]]),slate_ids=np.array([i]))
    baseline,_=development_data(data,out,c,'emer')
    with pytest.raises(FileNotFoundError):development_data(data,out,c,'emer_jev')
    def scores(ids):
        np.savez_compressed(data/'jev/scores.npz',probabilities=np.full((2,2,2),.5),entropy=np.ones((2,2,2)),
            confidence=np.zeros((2,2,2)),candidate_ids=ids,slate_ids=np.arange(2),model=np.array('test'),
            revision=np.array('fixed'),objective_names=np.array(['a','b']))
        atomic_write_json(data/'jev/complete.json',dict(complete=True,registration=dict(config_hash=canonical_hash(c)),
            scores_sha256=sha256_file(data/'jev/scores.npz')))
    scores(np.array([[0,1],[2,3]]));full,_=development_data(data,out,c,'emer_jev')
    for name in c['split_names']:
        for key in ('candidate_features','behavior_scores','labels','mask'):
            np.testing.assert_array_equal(baseline[name].arrays[key],full[name].arrays[key])
    scores(np.array([[1,0],[2,3]]))
    with pytest.raises(ValueError,match='Candidate alignment'):development_data(data,out,c,'emer_jev')
