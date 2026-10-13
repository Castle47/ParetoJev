"""Regression checks for weighted training and paired confirmation statistics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from paretojev.models.emer import emer_loss
from test_emer import batch


def runner():
    path=Path(__file__).resolve().parents[1]/'scripts/run_emer_jev.py'
    spec=importlib.util.spec_from_file_location('emer_runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_training_row_weights_equal_explicit_replication():
    b=batch();scores=torch.randn(2,4);w=torch.ones(2,7)
    normalized=torch.tensor([.5,1.5])
    weighted=emer_loss(scores,b,w,row_weights=normalized)['loss']
    fields={name:torch.repeat_interleave(getattr(b,name),torch.tensor([1,3]),dim=0)
        for name in ('candidate_features','behavior_scores','jev_scores','labels','preference_weights',
            'mask','jev_confidence','jev_entropy','popularity','history_length')}
    replicated=type(b)(**fields)
    expected=emer_loss(torch.repeat_interleave(scores,torch.tensor([1,3]),dim=0),replicated,torch.ones(4,7))['loss']
    torch.testing.assert_close(weighted,expected)


def test_identical_predictions_have_exactly_zero_paired_intervals():
    m=runner();b=batch()
    # Each slate has coverage for every positive goal, independently of scoring.
    b.labels[:,0,:6]=1
    data=SimpleNamespace(users=np.array([1,2]),sample_weights=np.array([1.,7.]),arrays={'mask':b.mask,'labels':b.labels})
    scores={f'{arm}/{seed}':np.array([[4.,3.,2.,1.],[4.,3.,2.,1.]])
        for arm in ('emer','emer_jev') for seed in (7,17,29)}
    c={'directions':[1,1,1,1,1,1,-1],'objectives':['is_click','long_view','is_like','is_follow','is_comment','is_forward','is_hate'],
        'training':{'seeds':[7,17,29]},'evaluation':{'top_k':3,'bootstrap_seed':1,'bootstrap_samples':30,
            'click_maximum_absolute_ndcg_drop':.005,'hate_maximum_absolute_risk_increase':.005}}
    result=m.paired_report(scores,data,c)
    for row in result['objectives'].values():
        assert row['absolute_gain']==0 and row['confidence_interval']==[0.,0.]
    assert result['positive_mean']['confidence_interval']==[0.,0.]
    assert not result['all_effect_gates']
