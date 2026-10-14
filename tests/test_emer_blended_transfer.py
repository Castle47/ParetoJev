import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from test_emer_balanced_transfer import unit_data
from run_emer_blended_transfer import blended_posterior,fit
from run_emer_balanced_transfer import active_goal_rates,balanced_posterior,admissible
from paretojev.models.emer import EMERConfig,EMERReranker,pairwise_losses
from paretojev.experiments.comparison import _save_checkpoint


def test_blended_loss_has_exact_endpoints_and_midpoint():
    y=torch.zeros(4,2,7);y[:,0,0]=1;y[0,0,1:]=1
    mask=torch.ones(4,2,dtype=torch.bool);weights=torch.tensor([.4,.8,1.2,1.6])
    scores=torch.tensor([[2.,0.],[0.,2.],[1.,0.],[-1.,0.]],requires_grad=True)
    rates=active_goal_rates(y,mask,weights)
    target=y[...,:-1].sum(-1)-y[...,-1]
    original=(pairwise_losses(scores,target[...,None],mask).squeeze(-1)*weights).mean()
    balanced=balanced_posterior(scores,y,mask,weights,rates)
    assert not torch.isclose(original,balanced)
    torch.testing.assert_close(blended_posterior(scores,y,mask,weights,rates,0),original)
    torch.testing.assert_close(blended_posterior(scores,y,mask,weights,rates,1),balanced)
    midpoint=blended_posterior(scores,y,mask,weights,rates,.5)
    torch.testing.assert_close(midpoint,(original+balanced)/2)
    midpoint.backward();assert torch.isfinite(scores.grad).all()
    with pytest.raises(ValueError):blended_posterior(scores,y,mask,weights,rates,1.1)


def test_both_blended_arms_keep_exact_initialization_and_guarded_selection(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(91)
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/emer_blended_transfer_v1.json').read_text())
    c['training']['epochs']=1;c['training']['batch_size']=4;c['directions']=[1]*6+[-1]
    anchor=EMERReranker(EMERConfig());d=unit_data(tmp_path);path=tmp_path/'anchor.pt'
    _save_checkpoint(path,dict(model_config=asdict(anchor.config),state=anchor.state_dict()))
    class ForbiddenOuter:
        def __getattribute__(self,name):raise AssertionError('Outer data accessed')
    for flag in (False,True):
        r=fit({'train':d,'validation':d,'test':ForbiddenOuter()},c,path,7,flag,tmp_path/f'blend-{flag}.pt')
        assert r['posterior_mix']==.5 and r['parameters']==43345 and r['initial_predictions_exact']
        assert len(r['history'])==1 and admissible(r['validation_objectives'],r['anchor_objectives'],c)
