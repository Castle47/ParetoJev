import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multiobjective_residual import objective,swap_weights
from jev_residual import rank_basis


def test_multigoal_gradient_and_weight_scale_invariance():
    rng=np.random.default_rng(123);base=rng.normal(size=(5,4));mask=np.ones((5,4),bool)
    y=rng.integers(0,2,size=(5,4,7));y[:,0,:]=1;y[:,1,:]=0
    w=np.array([1.,2.,1.,4.,3.]);pw=swap_weights(base,y,mask,w)
    np.testing.assert_allclose(pw,swap_weights(base,y,mask,5*w),atol=1e-15)
    x=rank_basis(rng.random((5,4,7)),mask);beta=rng.normal(size=7)*.1
    args=(base,x,mask,pw,.01,.25);_,analytic=objective(beta,*args);numeric=[]
    for i in range(7):
        delta=np.zeros(7);delta[i]=1e-5
        numeric.append((objective(beta+delta,*args)[0]-objective(beta-delta,*args)[0])/2e-5)
    np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=1e-8)


def test_equal_feedback_candidates_and_outside_top3_have_no_swap_weight():
    base=np.array([[5.,4.,3.,2.,1.]])
    mask=np.ones((1,5),bool);y=np.zeros((1,5,7));y[0,0,:]=1;y[0,3,:]=1
    pw=swap_weights(base,y,mask,np.ones(1))
    assert pw[0,0,3]==0 and pw[0,3,4]==0
