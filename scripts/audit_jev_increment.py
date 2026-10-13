#!/usr/bin/env python3
"""Exploratory signal audit on reused data; never claims new confirmation."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paretojev.evaluation.comparison import _contributions
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import sha256_file
from run_emer_jev import dataset,load_model,predict


def oriented(values):
    return np.concatenate((values[...,:6],1-values[...,6:]),-1)


def pair_statistics(scores,labels):
    """Per-slate binary AUC: each positive-negative pair, ties credited 1/2."""
    pairs=labels[:,:,None]>labels[:,None,:]
    difference=scores[:,:,None]-scores[:,None,:]
    count=pairs.sum((1,2))
    wins=((difference>0)+.5*(difference==0))*pairs
    auc=np.divide(wins.sum((1,2)),count,out=np.zeros(len(scores)),where=count>0)
    return auc,(count>0).astype(float),pairs,difference


def summarize(rows,weights,users,bootstrap=1000):
    """Common-support paired weighted ratios and user-cluster intervals."""
    groups,codes=np.unique(users,return_inverse=True)
    numerator=np.column_stack([row[0] for row in rows])*weights[:,None]
    denominator=np.column_stack([row[1] for row in rows])*weights[:,None]
    gn=np.zeros((len(groups),len(rows)));gd=np.zeros_like(gn)
    np.add.at(gn,codes,numerator);np.add.at(gd,codes,denominator)
    def value(count):
        num=count@gn;den=count@gd
        return np.divide(num,den,out=np.full(len(rows),np.nan),where=den>0)
    estimate=value(np.ones(len(groups)));rng=np.random.default_rng(20261013);samples=[]
    for _ in range(bootstrap):
        count=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))
        samples.append(value(count))
    samples=np.asarray(samples)
    delta=samples[:,1]-samples[:,0];valid=delta[np.isfinite(delta)]
    interval=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None
    return dict(values=estimate.tolist(),difference=float(estimate[1]-estimate[0]),
        difference_ci=interval,users=len(groups),valid_replicates=len(valid))


def audit_split(data,base_scores,names,bootstrap=1000):
    labels=oriented(data.arrays['labels'].numpy());behavior=oriented(data.arrays['behavior_scores'].numpy())
    jev=oriented(data.arrays['jev_scores'].numpy());mask=data.arrays['mask'].numpy()
    actual_labels=data.arrays['labels'].numpy();w=data.sample_weights;result={}
    for j,name in enumerate(names):
        b_auc,b_den,pairs,b_margin=pair_statistics(behavior[...,j],labels[...,j])
        j_auc,j_den,_,j_margin=pair_statistics(jev[...,j],labels[...,j])
        e_aucs=[pair_statistics(s,labels[...,j])[0] for s in base_scores]
        auc=summarize([(b_auc*b_den,b_den),(j_auc*j_den,j_den),
            (np.mean(e_aucs,0)*b_den,b_den)],w,data.users,bootstrap)
        number=pairs.sum((1,2));safe=np.maximum(number,1)
        rescue=((b_margin<0)&(j_margin>0)&pairs).sum((1,2))/safe
        harm=((b_margin>0)&(j_margin<0)&pairs).sum((1,2))/safe
        tied=((b_margin==0)&pairs).sum((1,2))/safe
        denominator=np.sum(w*b_den)
        complement=dict(behavior_wrong_jev_right=float(np.sum(w*rescue)/denominator),
            behavior_right_jev_wrong=float(np.sum(w*harm)/denominator),
            behavior_tie_fraction=float(np.sum(w*tied)/denominator),
            note='Weighted fractions of positive-negative pairs, averaged within each eligible slate; strict errors exclude ties. Not a served oracle policy.')
        # Same seven-goal metric definitions as the completed main experiment.
        def contributions(s):
            num,den,*_=_contributions(s,actual_labels,mask,np.array([1]*6+[-1]),3,np.zeros_like(mask))
            return num[:,j],den[:,j]
        b_num,b_d=contributions(behavior[...,j]);j_num,j_d=contributions(jev[...,j])
        e=[contributions(s)[0] for s in base_scores]
        ranking=summarize([(b_num,b_d),(j_num,j_d),(np.mean(e,0),b_d)],w,data.users,bootstrap)
        result[name]=dict(within_slate_auc=auc,top3_metric=ranking,complementarity=complement,
            eligible_pair_slates=int(b_den.sum()),positive_bits=int(actual_labels[...,j].sum()),
            column_order=['goal-specific behavior','goal-specific Jev','frozen EMER averaged over fitted seeds'])
    return result


def main(a):
    c=read_json('configs/emer_jev.json');selection=read_json(a.data/'selection.json');models=[]
    for seed in c['training']['seeds']:
        spec=selection['selected'][str(seed)]['emer'];path=a.data/spec['path']
        if sha256_file(path)!=spec['checkpoint_sha256']:raise ValueError('Frozen base changed')
        models.append(load_model(path))
    report=dict(status='exploratory reused data; April22 is development now, not a fresh test',splits={},
        protocol_findings={
            'long_view_question':'Existing question says long view but omits the duration<=18s completion / duration>18s at-least-18s rule.',
            'history':'Last ten exposures, not selected positive interactions; no feedback is supplied.',
            'choice':'Within-slate categorical distributions, not calibrated marginal event probabilities.',
            'availability':'Logs do not provide feedback completion/availability timestamps; cannot silently substitute clicked history.',
            'label_definition_source':'https://github.com/chongminggao/KuaiRand#data-descriptions',
            'causality':'These are hypotheses and protocol differences, not proven causes of failed uplift.'})
    for name,file in [('train','dev-train.npz'),('validation','dev-validation.npz'),('april22_development','test.npz')]:
        d=dataset(a.data/file);scores=[predict(m,d,32) for m in models]
        report['splits'][name]=dict(slates=len(d),input_sha256=sha256_file(a.data/file),
            metrics=audit_split(d,scores,c['objectives'],a.bootstrap))
        print(name,'long_view',report['splits'][name]['metrics']['long_view'],flush=True)
    atomic_write_json(a.output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();torch.set_num_threads(4);main(a)
