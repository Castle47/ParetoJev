import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from jev_residual import rank_basis,score_scale,corrected_scores,objective
from audit_jev_increment import pair_statistics


def test_pair_auc_ties_and_no_support():
    s=np.array([[2.,1.,0.],[1.,1.,1.],[3.,2.,1.]])
    y=np.array([[1.,0.,0.],[1.,0.,0.],[1.,1.,1.]])
    auc,eligible,*_=pair_statistics(s,y)
    np.testing.assert_array_equal(auc,[1.,.5,0.]);np.testing.assert_array_equal(eligible,[1.,1.,0.])


def test_zero_correction_exact_and_padding_does_not_set_scale():
    base=np.array([[.2,.5,-1e30]]);mask=np.array([[True,True,False]])
    p=np.array([[[.3],[.7],[.99]]]);x=rank_basis(p,mask)
    np.testing.assert_array_equal(corrected_scores(base,x,np.zeros(1),mask),base)
    np.testing.assert_allclose(score_scale(base,mask),[.15])
    output=corrected_scores(base,x,np.array([100.]),mask)
    assert np.max(np.abs((output-base)[mask]))<=.25*.15+1e-14
    assert output[0,2]==base[0,2]


def test_residual_objective_gradient_matches_finite_differences():
    rng=np.random.default_rng(2);base=rng.normal(size=(5,4));p=rng.random((5,4,3))
    mask=np.ones((5,4),dtype=bool);x=rank_basis(p,mask);y=np.array([[1,0,1,0]]*5)
    beta=np.array([.1,-.2,.3]);args=(base,x,y,mask,np.array([1.,2.,3.,4.,5.]),.01)
    _,gradient=objective(beta,*args);numerical=[]
    for i in range(3):
        offset=np.zeros(3);offset[i]=1e-5
        numerical.append((objective(beta+offset,*args)[0]-objective(beta-offset,*args)[0])/2e-5)
    np.testing.assert_allclose(gradient,numerical,rtol=1e-6,atol=1e-8)


def test_uniform_signal_is_exactly_no_correction_and_order_is_equivariant():
    rng=np.random.default_rng(4);base=rng.normal(size=(3,10));mask=np.ones((3,10),dtype=bool)
    x=rank_basis(np.full((3,10,1),.1),mask)
    np.testing.assert_array_equal(x,np.zeros_like(x))
    np.testing.assert_array_equal(corrected_scores(base,x,np.array([4.]),mask),base)
    p=rng.random((3,10,2));order=rng.permutation(10);beta=np.array([.4,-.3])
    before=corrected_scores(base,rank_basis(p,mask),beta,mask)
    after=corrected_scores(base[:,order],rank_basis(p[:,order],mask[:,order]),beta,mask[:,order])
    np.testing.assert_allclose(after,before[:,order],rtol=0,atol=1e-15)
