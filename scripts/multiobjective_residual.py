"""Fixed-baseline top3 swap weights for seven-goal bounded rank correction."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from jev_residual import score_scale


def swap_weights(base,labels,mask,weights):
    n,k=base.shape
    if labels.shape!=(n,k,7) or weights.shape!=(n,) or not np.isfinite(labels).all():
        raise ValueError('Expected seven valid feedback targets and slate weights')
    order=np.argsort(-np.where(mask,base,-np.inf),axis=1,kind='stable')
    positions=np.argsort(order,axis=1,kind='stable')
    discount=np.zeros(k);discount[:min(3,k)]=1/np.log2(np.arange(2,min(3,k)+2))
    total=np.zeros((n,k,k),dtype='float64')
    pair_mask=mask[:,:,None]&mask[:,None,:]
    for goal in range(7):
        y=labels[...,goal] if goal<6 else 1-labels[...,goal]
        if goal<6:
            ideal=(np.sort(np.where(mask,y,0),axis=1)[:,::-1]*discount).sum(1)
            normalizer=np.sum(weights*(ideal>0))
            per_item_discount=discount[positions]/np.maximum(ideal[:,None],1e-12)
        else:
            normalizer=weights.sum()
            per_item_discount=(positions<3)/mask.sum(1).clip(1,3)[:,None]
        if normalizer<=0:raise ValueError('Missing goal support')
        delta=np.abs(per_item_discount[:,:,None]-per_item_discount[:,None,:])
        pairs=(y[:,:,None]>y[:,None,:])&pair_mask
        total+=weights[:,None,None]/normalizer*pairs*delta/7
    return total


def objective(beta,base,x,mask,pair_weights,ridge,cap):
    z=np.where(mask,base,0)/score_scale(base,mask)[:,None]+cap*np.tanh(x@beta)
    margin=z[:,:,None]-z[:,None,:]
    loss=np.sum(np.logaddexp(0,-margin)*pair_weights)+ridge*np.dot(beta,beta)
    d=-expit(-margin)*pair_weights
    ds=d.sum(2)-d.sum(1)
    grad=np.einsum('nk,nk,nkm->m',ds,cap*(1-np.tanh(x@beta)**2),x)+2*ridge*beta
    return float(loss),grad


def fit_correction(base,x,labels,mask,weights,ridge,cap=.25):
    pair_weights=swap_weights(base,labels,mask,weights)
    args=(base,x,mask,pair_weights,ridge,cap)
    result=minimize(objective,np.zeros(x.shape[-1]),args=args,method='L-BFGS-B',jac=True,
        bounds=[(-4,4)]*x.shape[-1],options={'maxiter':100,'ftol':1e-12})
    if not result.success:raise ValueError('Optimizer failed: '+str(result.message))
    return result.x,dict(iterations=int(result.nit),loss=float(result.fun),success=True,
        criterion='Seven-goal fixed-baseline top3 swap-weighted logistic loss; not exact NDCG maximization')
