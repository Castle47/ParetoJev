import json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_emer_balanced_transfer import warm_start,active_goal_rates,balanced_posterior,admissible,best_epoch,fit
from run_emer_jev import dataset,predict,load_model
from paretojev.models.emer import EMERConfig,EMERReranker
from paretojev.experiments.comparison import _save_checkpoint


def config():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/emer_balanced_transfer_v1.json').read_text())
    c['directions']=[1]*6+[-1]
    return c


def test_conditional_task_loss_equalizes_rare_and_frequent_support():
    labels=torch.zeros(4,2,7);labels[:,0,0]=1;labels[0,0,1:]=1
    mask=torch.ones(4,2,dtype=torch.bool);w=np.array([1.,2.,3.,4.])
    rates=active_goal_rates(labels,mask,w)
    torch.testing.assert_close(rates,torch.tensor([1.]+[.1]*6,dtype=torch.float64))
    scores=torch.zeros(4,2,requires_grad=True)
    loss=balanced_posterior(scores,labels,mask,torch.tensor(w/w.mean(),dtype=torch.float32),rates)
    assert float(loss.detach())==pytest.approx(np.log(2))
    loss.backward();assert torch.isfinite(scores.grad).all()
    labels[...,6]=0;rates=active_goal_rates(labels,mask,w)
    loss=balanced_posterior(scores,labels,mask,torch.tensor(w/w.mean(),dtype=torch.float32),rates)
    assert float(loss.detach())==pytest.approx(np.log(2)*6/7)


def test_selection_rejects_tradeoffs_and_preserves_earliest_tie():
    c=config();anchor=[.5]*6+[.99]
    poor=[.6,.6,.6,.49,.6,.6,.99];good=[.51]*6+[.99];hate=[.6]*6+[.98]
    rows=[dict(epoch=i+1,objectives=p,validation=float(np.mean(p[:6]))) for i,p in enumerate([poor,good,good,hate])]
    assert not admissible(poor,anchor,c) and not admissible(hate,anchor,c)
    assert best_epoch(anchor,rows,c)==2
    assert best_epoch(anchor,[rows[0]],c)==0
    assert not admissible([float('nan')]*7,anchor,c)


def unit_data(tmp_path):
    rng=np.random.default_rng(17);n,k=4,10
    labels=rng.integers(0,2,(n,k,7)).astype('float32');labels[:,0,:]=1;labels[:,1,:]=0
    path=tmp_path/'data.npz'
    np.savez(path,candidate_features=rng.normal(size=(n,k,65)).astype('float32'),
        behavior_scores=rng.random((n,k,7)).astype('float32'),jev_scores=rng.random((n,k,7)).astype('float32'),
        jev_entropy=np.ones((n,k,7),dtype='float32'),jev_confidence=rng.random((n,k,7)).astype('float32'),
        labels=labels,mask=np.ones((n,k),bool),history_length=np.ones((n,1),dtype='float32'),
        popularity=np.ones((n,k,1),dtype='float32'),slate_weights=np.ones(n),user_ids=np.arange(n))
    return dataset(path)


def test_identical_anchor_initialization_and_both_training_arms(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(9)
    anchor=EMERReranker(EMERConfig()).eval();d=unit_data(tmp_path)
    path=tmp_path/'anchor.pt';_save_checkpoint(path,dict(model_config=asdict(anchor.config),state=anchor.state_dict()))
    expected=predict(anchor,d,4);c=config();c['training']['epochs']=1;c['training']['batch_size']=4
    class ForbiddenOuter:
        def __getattribute__(self,name):raise AssertionError('Outer data accessed during fit')
    for flag in (False,True):
        model=warm_start(anchor,flag)
        np.testing.assert_array_equal(expected,predict(model,d,4))
        assert sum(p.numel() for p in model.parameters())==43345
        output=tmp_path/f'fit-{flag}.pt'
        r=fit({'train':d,'validation':d,'test':ForbiddenOuter()},c,path,7,flag,output)
        assert len(r['history'])==1 and r['initial_predictions_exact']
        assert r['best_epoch']==best_epoch(r['anchor_objectives'],r['history'],c)
        assert admissible(r['validation_objectives'],r['anchor_objectives'],c)
        assert load_model(output).config.use_jev==flag
