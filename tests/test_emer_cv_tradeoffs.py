import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_emer_cv_tradeoffs import support_statistics,weighted_pair_auc


def test_pair_auc_half_ties_masks_and_unsupported_slates():
    scores=np.array([[3,1,99],[2,2,99],[0,1,99],[8,9,99]],float)
    labels=np.array([[1,0,0],[1,0,1],[1,0,1],[0,0,1]],float)
    mask=np.tile([True,True,False],(4,1));weights=np.array([1,2,1,100],float)
    assert weighted_pair_auc(scores,labels,mask,weights)==pytest.approx(.5)
    assert weighted_pair_auc(scores,np.zeros_like(labels),mask,weights) is None


def test_support_counts_masked_labels_and_aggregates_user_weights():
    labels=np.array([[[1],[0],[1]],[[1],[0],[1]]],float)
    mask=np.array([[True,True,False],[True,False,False]])
    row=support_statistics(labels,mask,np.array([2.,1.]),np.array([7,7]))[0]
    assert row['positive_exposures']==2 and row['positive_slates']==2 and row['positive_users']==1
    assert row['weighted_prevalence']==pytest.approx(3/5)
    assert row['weighted_positive_slate_fraction']==1
    assert row['weighted_pair_supervision_fraction']==pytest.approx(2/3)
    assert row['positive_support_user_weight_ess']==1
