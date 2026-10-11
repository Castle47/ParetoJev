import numpy as np
from scipy.special import expit
from paretojev.training.behavior_calibration import fit_calibration, apply_calibration, cluster_logloss_difference
from paretojev.evaluation.behavior_metrics import metrics


def specification():
    return dict(slope_log_bounds=[-2,2],intercept_bounds=[-4,4],slope_penalty=.0001,
                intercept_penalty=.00001,max_iterations=200)


def test_separate_calibration_corrects_shift_without_changing_order():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(6000,7));truth=rng.binomial(1,expit(x)).astype('uint8')
    parameters=fit_calibration(truth,expit(x-1),np.ones(len(x)),specification())
    xe=rng.normal(size=(6000,7));ye=rng.binomial(1,expit(xe)).astype('uint8')
    raw=expit(xe-1);corrected=apply_calibration(raw,parameters)
    before=metrics(ye,raw,np.ones(len(xe)));after=metrics(ye,corrected,np.ones(len(xe)))
    for key in before:
        assert after[key]['logloss'] < before[key]['logloss']-.03
        assert abs(after[key]['auc']-before[key]['auc']) < 1e-12
    assert all(p['slope']>0 for p in parameters)


def test_unsigned_labels_do_not_wrap_in_logloss():
    y=np.array([[0]*7,[1]*7],dtype='uint8')
    result=metrics(y,np.full((2,7),.5),np.ones(2))
    assert all(abs(v['logloss']-np.log(2))<1e-12 for v in result.values())


def test_bootstrap_sign_and_identical_predictions():
    y=np.array([[0]*7,[1]*7]*20,dtype='uint8');w=np.ones(40);users=np.repeat(np.arange(10),4)
    better=np.where(y==1,.8,.2);worse=np.full_like(better,.5)
    r=cluster_logloss_difference(y,better,worse,w,users,replicates=100,seed=5)
    assert max(r['upper'])<0
    r=cluster_logloss_difference(y,better,better,w,users,replicates=100,seed=5)
    assert r['point']==r['lower']==r['upper']==[0.]*7
