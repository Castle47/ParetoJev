import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from develop_multiobjective_ensemble import ensemble


def test_ensemble_positive_scale_shift_invariance_and_permutation():
    rng=np.random.default_rng(42);x=rng.normal(size=(3,4,5));mask=np.ones((4,5),bool)
    for recipe in ('equal_weight_three_seed_zscore','equal_weight_three_seed_rank'):
        y=ensemble(x,mask,recipe)
        changed=x*np.array([2.,3.,4.])[:,None,None]+np.array([10.,-4.,12.])[:,None,None]
        np.testing.assert_allclose(ensemble(changed,mask,recipe),y,atol=1e-14,rtol=0)
        p=np.array([3,0,4,2,1]);np.testing.assert_allclose(ensemble(x[:,:,p],mask[:,p],recipe),y[:,p],atol=1e-14,rtol=0)


def test_uniform_constituents_give_zero_and_ignore_padding():
    x=np.ones((3,2,4));mask=np.array([[True,True,True,False],[True,True,True,False]])
    x[:,:,3]=999
    for recipe in ('equal_weight_three_seed_zscore','equal_weight_three_seed_rank'):
        np.testing.assert_array_equal(ensemble(x,mask,recipe),np.zeros((2,4)))
