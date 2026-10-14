#!/usr/bin/env python3
"""Export audited reranker weights, per-fold encoders and label-free inference."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
from paretojev.data.kuairand_features_v2 import CORE_FIELDS,EXTRA_FIELDS
from paretojev.experiments.comparison import _source_identity
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file
from run_emer_scale import immutable_json
from run_emer_user_cv import common_arrays,user_bucket
from run_emer_jev import load_model


def inference_inputs(common,raw_features):
    return dict(raw_features=raw_features,behavior_scores=common['behavior_scores'],jev_scores=common['jev_scores'],
        jev_entropy=common['jev_entropy'],jev_confidence=common['jev_confidence'],user_ids=common['user_ids'],
        candidate_ids=common['candidate_ids'],mask=common['mask'],popularity=common['popularity'],history_length=common['history_length'])


def export(a):
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use a fresh output directory; published bundles are immutable')
    policy_root=a.development/'matched-fallback-v1';run=policy_root/'blended';cv=a.development/'user-cv-v1'
    root_reg=read_json(policy_root/'registration.json')
    for p,h in root_reg['inputs'].items():
        if sha256_file(p)!=h:raise ValueError('Frozen policy source differs')
    r=read_json(run/'evaluation.json');rep=read_json(run/'replay.json');audit=read_json(run/'audit.json')
    if not r['screen_passed'] or not rep['exact_predictions_and_metrics'] or not audit['policy_replayed']:raise ValueError('Audited passing exploratory candidate required')
    for n,key in [('evaluation.json','evaluation_sha256'),('predictions.npz','predictions_sha256')]:
        if sha256_file(run/n)!=rep[key]:raise ValueError('Policy replay changed')
    if audit['selection_sha256']!=sha256_file(run/'selection.json'):raise ValueError('Selection changed')
    original=read_json(cv/'registration.json');c=original['config']
    for p,h in original['files'].items():
        if sha256_file(p)!=h:raise ValueError('Original data/source changed')
    identity=_source_identity()
    if identity['source_sha256']!=original['model_source_sha256']:raise ValueError('Model/encoder source changed')
    a.output.mkdir(parents=True,exist_ok=True)
    def put(source,relative):
        dest=a.output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha256_file(dest)!=sha256_file(source):raise ValueError('Existing export file differs; use a new output')
        if not dest.exists():shutil.copy2(source,dest)
        return relative
    selected=read_json(run/'selection.json')['folds'];anchors=read_json(cv/'selection.json')['folds']
    decisions={(str(d['fold']),str(d['seed'])):d for d in read_json(run/'policy.json')['decisions']}
    routes={}
    for f in range(3):
        folder=cv/f'fold-{f}';manifest=read_json(folder/'data_manifest.json')
        if sha256_file(folder/'feature_spec.json')!=manifest['files']['feature_spec.json']:raise ValueError('Fold encoder changed')
        route=dict(feature_spec=put(folder/'feature_spec.json',f'encoders/fold-{f}.json'),seeds={})
        for seed in (7,17,29):
            choices=dict(selected[str(f)][str(seed)])
            choices['anchor']=dict(anchors[str(f)][str(seed)]['emer'],path=str(folder/anchors[str(f)][str(seed)]['emer']['path']))
            route['seeds'][str(seed)]={}
            for arm,spec in choices.items():
                p=Path(spec['path']);h=sha256_file(p)
                if h!=spec['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                model=load_model(p);expected=arm=='emer_jev' and decisions[(str(f),str(seed))]['use_jev']
                if model.config.use_jev!=expected or sum(t.numel() for t in model.parameters())!=43345:raise ValueError('Wrong model arm or capacity')
                route['seeds'][str(seed)][arm]=dict(path=put(p,f'models/{h}.pt'),sha256=h,use_jev=expected,best_epoch=spec['best_epoch'])
        routes[str(f)]=route
    repo=Path(__file__).resolve().parents[1]
    for p in sorted((repo/'src/paretojev').rglob('*.py')):put(p,'python/paretojev/'+p.relative_to(repo/'src/paretojev').as_posix())
    for name in ('candidate_runtime.py','infer.py'):put(repo/'deployment/emer_candidate'/name,name)
    for name in ('registration.json','selection.json','policy.json','audit.json','evaluation.json','replay.json'):put(run/name,'evidence/'+name)
    put(policy_root/'registration.json','evidence/root_policy_registration.json')
    for name in ('LICENSE','pyproject.toml'):
        if (repo/name).exists():put(repo/name,name)
    common,raw=common_arrays(SimpleNamespace(data=a.development));routes_all=np.array([user_bucket(u,'outer',c['partition_seed'],3) for u in common['user_ids']])
    ids=np.concatenate([np.flatnonzero(routes_all==f)[:2] for f in range(3)])
    np.savez_compressed(a.output/'example_input.npz',**{k:v[ids] for k,v in inference_inputs(common,raw).items()})
    schema=dict(raw_feature_names=CORE_FIELDS+EXTRA_FIELDS,raw_features='[N,10,26]; numeric NaN/nonfinite values use the frozen missing-value encoding; tab must be integer0..14',
        behavior_scores='[N,10,7] fixed HGB+calibration marginal probabilities',jev_scores='[N,10,7] native JEV-27B categorical choice probabilities, sum across candidates=1',
        jev_entropy='[N,10,7] normalized entropy repeated across candidates',jev_confidence='[N,10,7] one minus entropy',
        user_ids='[N] nonnegative integers; same ID must keep same meaning',candidate_ids='[N,10] distinct nonnegative integer IDs per slate',
        popularity='[N,10] or[N,10,1] frozen prior exposure counts',history_length='[N] or[N,1] strict-past count',mask='[N,10] all True for this frozen protocol',
        forbidden='labels, feedback, pre-encoded candidate_features or any undeclared fields',outputs='scores[N,10], stable descending ranking_indices[N,10], ranked_candidate_ids[N,10], fold_ids[N], uses_jev[N]')
    atomic_write_json(a.output/'input_schema.json',schema)
    packages={p:importlib.metadata.version(p) for p in ('numpy','torch','threadpoolctl','pandas','scikit-learn','scipy','joblib','requests','PyYAML','tqdm')}
    (a.output/'requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in packages.items())+'\n')
    (a.output/'README.md').write_text('''# Frozen EMER+Jev matched-fallback candidate

This is the reranker package for the exploratory36K result (+1.97% mean six-goal NDCG@3 versus the matched EMER). Independent Apr23-25 confirmation is still missing. Long-view declines0.66%; not every goal improves.

Install the recorded dependencies using Python3.12 (reference3.12.11, macOS arm64). The bundle includes frozen model/encoder Python source and23 or fewer deduplicated checkpoints; it does not include the upstream HGB or27B Jev model. Supply their already-computed numeric inputs. No API is called by this inference program. Cross-hardware numerical identity is not promised; exact replay is checked on the recorded reference environment.

Run from this directory:

```sh
python -m pip install -r requirements-lock.txt
python infer.py --input example_input.npz --output /tmp/emer-example-output.npz --arm emer_jev --seed 7
```

Arms: `emer` is the matched mixed-loss control, `emer_jev` is the inner-selected Jev-or-matched-baseline policy, `anchor` is the original same-fold EMER. Seeds7/17/29 are all preserved; this is not a seed ensemble. Scores are ranking logits, not probabilities. The first3 ranked candidate IDs form the evaluated top3.

The runtime routes each user by the original SHA256 hash and applies that fold's training-only feature specification. Never standardize inputs again or provide the old global65-dimensional encoding. See input_schema.json for the exact26 raw features and seven-goal inputs. Feedback labels are rejected; internal placeholder labels exist only to satisfy the historical tensor shape and are not read by the model.

example_input.npz contains6 real label-free development slates covering all3 routes; it is an API example, not a new benchmark. Manifest/model/source hashes are verified on load. The evidence directory preserves policy choices, selected checkpoint identities, all goals and exact-replay receipts. Absolute paths within historical evidence are provenance only; runtime paths are relative.

Feature construction, behavior prediction, Jev production, cache lookup and end-to-end serving latency are outside this reranker-only executable. The existing public-data source and original source audits remain in the project.
''')
    files={p.relative_to(a.output).as_posix():sha256_file(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('manifest.json','manifest.sha256')}
    source=read_json(a.development/'jev-native-v9-fp16/registration.json')['source'];common_manifest=read_json(a.development/'training/common_manifest.json')
    manifest=dict(candidate_id='emer-matched-blended-v1',scope='Exploratory routed reranker candidate; independent confirmation pending',
        model_source_sha256=identity['source_sha256'],seeds=[7,17,29],routing=dict(namespace='outer',partition_seed=c['partition_seed'],folds=3,hash='SHA256 first8bytes big-endian modulo3'),
        routes=routes,objectives=c['objectives'],directions=c['directions'],runtime=dict(python=platform.python_version(),machine=platform.machine(),packages=packages,device='cpu',threads=2,batch_size=32),
        upstream=dict(behavior_checkpoint_sha256=common_manifest['identity']['behavior_sha256'],jev_model=source['model'],jev_revision=source['huggingface_revision'],jev_producer_id=source['producer_id']),
        provenance=dict(policy_registration_sha256=sha256_file(policy_root/'registration.json'),policy_evaluation_sha256=sha256_file(run/'evaluation.json'),
            original_cv_registration_sha256=sha256_file(cv/'registration.json'),export_script_sha256=sha256_file(Path(__file__))),files=files)
    immutable_json(a.output/'manifest.json',manifest)
    (a.output/'manifest.sha256').write_text(sha256_file(a.output/'manifest.json')+'  manifest.json\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(a.output.rglob('*')):
            if p.is_file():
                info=zipfile.ZipInfo(a.output.name+'/'+p.relative_to(a.output).as_posix(),date_time=(2026,10,15,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,p.read_bytes())
    print(json.dumps(dict(bundle=str(a.output),archive=str(archive),archive_sha256=sha256_file(archive),manifest_sha256=sha256_file(a.output/'manifest.json'),files=len(files)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);export(p.parse_args())
