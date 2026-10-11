"""Monotone, weighted calibration fitted only on an explicitly separate period."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def fit_calibration(y, predictions, weights, config):
    z=logit(np.clip(np.asarray(predictions,dtype='float64'),1e-7,1-1e-7))
    w=np.asarray(weights,dtype='float64');w=w/w.sum()
    result=[]
    for j in range(y.shape[1]):
        truth=np.asarray(y[:,j],dtype='float64');score=z[:,j]
        if not 0 < truth.sum() < len(truth): raise ValueError('Calibration needs both classes per task')
        pa,pb=config['slope_penalty'],config['intercept_penalty']
        def objective(theta):
            a=np.exp(theta[0]);b=theta[1];v=a*score+b
            residual=expit(v)-truth
            value=np.sum(w*(np.logaddexp(0,v)-truth*v))+pa*(a-1)**2+pb*b*b
            grad=np.array([a*np.sum(w*residual*score)+2*pa*(a-1)*a,
                           np.sum(w*residual)+2*pb*b])
            return value,grad
        fitted=minimize(objective,[0.,0.],jac=True,method='L-BFGS-B',
            bounds=[config['slope_log_bounds'],config['intercept_bounds']],
            options={'maxiter':config['max_iterations'],'ftol':1e-12,'gtol':1e-8})
        if not fitted.success or not np.isfinite(fitted.fun):
            raise ValueError(f'Calibration failed for task {j}: {fitted.message}')
        result.append(dict(slope=float(np.exp(fitted.x[0])),intercept=float(fitted.x[1]),
            objective=float(fitted.fun),iterations=int(fitted.nit),status=str(fitted.message)))
    return result


def apply_calibration(predictions, parameters):
    z=logit(np.clip(np.asarray(predictions,dtype='float64'),1e-7,1-1e-7))
    if z.shape[1]!=len(parameters): raise ValueError('Task order/width mismatch')
    return expit(z*np.array([p['slope'] for p in parameters])+np.array([p['intercept'] for p in parameters]))


def cluster_logloss_difference(y, first, second, weights, users, *, replicates, seed):
    """First minus second; resample user clusters, conditional on fitted models."""
    y=np.asarray(y,dtype='float64');w=np.asarray(weights,dtype='float64')
    def losses(p):
        p=np.clip(p,1e-7,1-1e-7)
        return -y*np.log(p)-(1-y)*np.log1p(-p)
    _,inverse=np.unique(users,return_inverse=True);n=int(inverse.max())+1
    sums=np.zeros((n,y.shape[1]));den=np.zeros(n)
    np.add.at(sums,inverse,(losses(first)-losses(second))*w[:,None]);np.add.at(den,inverse,w)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        idx=rng.integers(0,n,n);values.append(sums[idx].sum(axis=0)/den[idx].sum())
    values=np.asarray(values)
    return {'point':(sums.sum(axis=0)/den.sum()).tolist(),
            'lower':np.quantile(values,.025,axis=0).tolist(),'upper':np.quantile(values,.975,axis=0).tolist(),
            'users':n,'replicates':replicates,'interpretation':'negative favors first; exploratory, no multiplicity correction'}
