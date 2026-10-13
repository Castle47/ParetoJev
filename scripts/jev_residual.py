"""Small bounded Jev-only correction; frozen baseline scores are never fitted."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def rank_basis(probabilities,mask):
    p=np.asarray(probabilities,dtype='float64')
    if p.ndim==2:p=p[...,None]
    valid=mask[:,:,None,None]&mask[:,None,:,None]
    better=(p[:,None,:,:]>p[:,:,None,:])&valid
    equal=(p[:,None,:,:]==p[:,:,None,:])&valid
    rank=1+better.sum(2)+.5*np.maximum(equal.sum(2)-1,0)
    rank=rank/mask.sum(1).clip(1)[:,None,None]
    mean=(rank*mask[...,None]).sum(1)/mask.sum(1).clip(1)[:,None]
    centered=np.where(mask[...,None],mean[:,None,:]-rank,0)
    # A uniform signal has exactly no ranking information. Avoid a tiny common
    # offset from floating-point averaging of tied ranks.
    strict=better.any((1,2))
    return np.where(strict[:,None,:],centered,0)


def score_scale(base,mask):
    safe=np.where(mask,base,0).astype('float64');count=mask.sum(1).clip(1)
    mean=safe.sum(1)/count
    variance=(np.where(mask,safe-mean[:,None],0)**2).sum(1)/count
    return np.sqrt(variance).clip(1e-6)


def corrected_scores(base,x,beta,mask,cap=.25):
    delta=cap*score_scale(base,mask)[:,None]*np.tanh(x@beta)
    return np.where(mask,base+delta,base)


def objective(beta,base,x,y,mask,weights,ridge,cap=.25):
    scale=score_scale(base,mask);safe=np.where(mask,base,0)
    z=safe/scale[:,None]+cap*np.tanh(x@beta)
    pairs=(y[:,:,None]>y[:,None,:])&(mask[:,:,None]&mask[:,None,:])
    count=pairs.sum((1,2));eligible=count>0
    row_weights=np.where(eligible,weights,0);row_weights=row_weights/max(row_weights.sum(),1e-12)
    pair_weights=row_weights[:,None,None]*pairs/count.clip(1)[:,None,None]
    difference=z[:,:,None]-z[:,None,:]
    loss=np.sum(np.logaddexp(0,-difference)*pair_weights)+ridge*np.dot(beta,beta)
    dloss=-expit(-difference)*pair_weights
    dscore=dloss.sum(2)-dloss.sum(1)
    slope=cap*(1-np.tanh(x@beta)**2)
    gradient=np.einsum('nk,nk,nkm->m',dscore,slope,x)+2*ridge*beta
    return float(loss),gradient


def fit_correction(base,x,y,mask,weights,ridge,cap=.25):
    result=minimize(objective,np.zeros(x.shape[-1]),args=(base,x,y,mask,weights,ridge,cap),
        method='L-BFGS-B',jac=True,bounds=[(-4,4)]*x.shape[-1],options={'maxiter':100,'ftol':1e-12})
    if not result.success:raise ValueError('Correction optimizer failed: '+result.message)
    return result.x,dict(iterations=int(result.nit),loss=float(result.fun),success=True)
