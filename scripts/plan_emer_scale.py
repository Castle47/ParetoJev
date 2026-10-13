#!/usr/bin/env python3
"""Pilot-based precision planning; does not open any new test observations."""
import argparse
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from paretojev.evaluation.comparison import _contributions
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import sha256_file


def paired_precision(base_num, full_num, denominator, weights, users):
    """Delta-method SE of paired weighted ratios, clustering all rows of a user.

    Numerators have already been averaged across the same fixed trained seeds.
    This is a superpopulation user-cluster approximation, not a stratified finite
    population survey variance, nor uncertainty over training seeds or future days.
    """
    b, f, d = (np.asarray(x, dtype=float) for x in (base_num, full_num, denominator))
    w = np.asarray(weights, dtype=float)
    if b.shape != f.shape or b.shape != d.shape or b.ndim != 2:
        raise ValueError('Statistics must have matching [slate, objective] shapes')
    if w.shape != (len(b),) or len(users) != len(b) or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Positive finite row weights and matching users required')
    if not all(np.isfinite(x).all() for x in (b,f,d)) or (d<0).any():
        raise ValueError('Invalid statistics')
    groups,codes=np.unique(users,return_inverse=True)
    if len(groups)<2: raise ValueError('At least two user clusters required')
    totals=(d*w[:,None]).sum(0)
    if (totals<=0).any(): raise ValueError('Missing goal support')
    base=(b*w[:,None]).sum(0)/totals
    delta=((f-b)*w[:,None]).sum(0)/totals
    influence=w[:,None]*((f-b)-d*delta)/totals
    clustered=np.zeros((len(groups),b.shape[1]));np.add.at(clustered,codes,influence)
    se=np.sqrt(len(groups)/(len(groups)-1)*(clustered**2).sum(0))
    return base,delta,se


def projected_slates(pilot_n, standard_error, effect, power=.8, alpha=.05):
    if pilot_n<2 or standard_error<0 or effect<=0 or not 0<power<1 or not 0<alpha<1:
        raise ValueError('Invalid planning inputs')
    if standard_error==0:
        return None  # identical pilot rankings do not establish zero future noise
    factor=norm.ppf(1-alpha/2)+norm.ppf(power)
    return max(2,math.ceil(pilot_n*(factor*standard_error/effect)**2))


def main(a):
    c=read_json('configs/emer_jev.json')
    with np.load(a.pilot/'test.npz') as z: data={k:z[k] for k in z.files}
    with np.load(a.pilot/'predictions.npz') as z: scores={k:z[k] for k in z.files}
    mask=data['mask'];labels=data['labels'];stats={}
    for arm in ('emer','emer_jev'):
        parts=[_contributions(scores[f'{arm}/{s}'],labels,mask,np.array(c['directions']),3,
            np.zeros_like(mask))[:2] for s in c['training']['seeds']]
        den=parts[0][1]
        if any(not np.array_equal(p[1],den) for p in parts):raise ValueError('Seed support differs')
        stats[arm]=np.mean([p[0] for p in parts],axis=0)
    w=data['slate_weights'];users=data['user_ids'];n=len(w);g=len(np.unique(users))
    base,delta,se=paired_precision(stats['emer'],stats['emer_jev'],den,w,users)
    goals={}
    for j,name in enumerate(c['objectives']):
        eligible=den[:,j]>0; ew=w[eligible]
        row=dict(pilot_baseline=float(base[j]),pilot_difference=float(delta[j]),
            paired_user_cluster_standard_error=float(se[j]),
            eligible_slates=int(eligible.sum()),eligible_users=len(np.unique(users[eligible])),
            observed_positive_labels=int(labels[...,j].sum()),
            positive_label_users=len(np.unique(users[(labels[...,j].sum(1)>0)])),
            kish_effective_eligible_slates=float(ew.sum()**2/(ew@ew)),
            note='Kish describes weight imbalance only, not independent observations.')
        if j<6:
            row['planning_relative_effects']={str(effect):dict(absolute_effect=float(base[j]*effect),
                iid_style_required_slates=projected_slates(n,se[j],base[j]*effect)) for effect in (.01,.02,.03)}
        else:
            row['planning_absolute_risk_reduction']=dict(effect=.00005,
                iid_style_required_slates=projected_slates(n,se[j],.00005),
                note='0.005 percentage points risk reduction; never a 1% relative increase of near-one no-hate utility.')
        goals[name]=row
    scale=read_json(a.config);train=sum(scale['sample_budgets']['train']);val=sum(scale['sample_budgets']['validation'])
    test=scale['evaluation']['test_target_slates'];calls=(train+val+test)*7
    factor=norm.ppf(.975)+norm.ppf(.8);sensitivity=[]
    for target_n in (20000,50000):
        for target_users in (10000,20000,27285):
            if target_users>target_n:continue
            for rho in (0,.1,.3,1):
                # Exchangeable equal-sized clusters are a sensitivity model,
                # not an ICC estimate from only 22 extra pilot rows.
                inflation=(1+(target_n/target_users-1)*rho)/(1+(n/g-1)*rho)
                new_se=se[1]*math.sqrt(n/target_n*inflation)
                sensitivity.append(dict(slates=target_n,users=target_users,assumed_within_user_correlation=rho,
                    long_view_relative_mde_80pct=float(factor*new_se/base[1])))
    out=dict(role='planning from already-opened April22 pilot, no new-test evaluation',
        pilot=dict(slates=n,users=g,seeds=c['training']['seeds'],
            tensor_sha256=sha256_file(a.pilot/'test.npz'),predictions_sha256=sha256_file(a.pilot/'predictions.npz')),
        assumptions=dict(power=.8,two_sided_alpha=.05,primary='long_view NDCG@3',
            target_relative_gain=.01,independent_user_universe_ceiling=27285,
            projection='SE scales as sqrt(pilot_n/new_n) only if weighting, sampling and correlation design remain comparable.',
            caveats=['Re-trained rankings and new dates may change variance.',
                'Seven effects do not each have 80% power; only the primary is sized.',
                'No joint power guarantee for all success gates.',
                'Day generalization is not measured by user-only clusters.',
                'Only 578 pilot users and 22 extra rows: ICC cannot be reliably estimated.',
                'Finite 27K users limit independent-user growth; more rows alone do not imply target power.']),
        goals=goals,cluster_sensitivity=sensitivity,
        budget=dict(train_slates=train,validation_slates=val,planned_test_slates=test,
            nominal_requests=calls,maximum_attempts_with_10pct_retry_budget=math.ceil(calls*1.1),
            minimum_start_interval_seconds=1,nominal_spacing_floor_hours=(calls-1)/3600,
            note='Excludes final response latency, failures, downtime and training; not a throughput commitment.',
            local_memory_bytes=8589934592,model_parameter_count=26895998464,
            theoretical_bf16_weight_bytes=26895998464*2,
            theoretical_4bit_weight_bytes=26895998464/2,
            local_full_model_fit=False))
    a.output.mkdir(parents=True,exist_ok=True);atomic_write_json(a.output/'power_plan.json',out)
    print('long-view 1% planning:',goals['long_view']['planning_relative_effects']['0.01'],flush=True)
    print('budget:',out['budget'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev_scale.json'))
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
