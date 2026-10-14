"""Label-free CPU inference for the frozen user-routed matched-EMER candidate."""
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import paretojev
from paretojev.data.emer import encode_features
from paretojev.data.schema import SlateBatch
from paretojev.models.emer import EMERConfig,EMERReranker

REQUIRED={'raw_features','behavior_scores','jev_scores','jev_entropy','jev_confidence',
    'user_ids','candidate_ids','popularity','history_length','mask'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def user_routes(users,partition_seed=2026101407,folds=3):
    users=np.asarray(users)
    if users.ndim!=1 or users.dtype.kind not in 'iu' or (users<0).any():raise ValueError('Integer nonnegative user IDs required')
    return np.array([int.from_bytes(hashlib.sha256(f'outer|{partition_seed}|{int(u)}'.encode()).digest()[:8],'big')%folds for u in users],dtype='int8')


def validate_inputs(values):
    if set(values)!=REQUIRED:raise ValueError(f'Exact label-free schema required; missing={sorted(REQUIRED-set(values))}, unexpected={sorted(set(values)-REQUIRED)}')
    x={k:np.asarray(v) for k,v in values.items()};n=len(x['user_ids'])
    if n==0:raise ValueError('At least one slate required')
    if x['raw_features'].shape!=(n,10,26):raise ValueError('raw_features must be[N,10,26], not pre-encoded features')
    tab=x['raw_features'][...,0]
    if not np.isfinite(tab).all() or (tab!=np.floor(tab)).any() or (tab<0).any() or (tab>=15).any():raise ValueError('Invalid tab IDs')
    if x['candidate_ids'].shape!=(n,10) or x['candidate_ids'].dtype.kind not in 'iu' or (x['candidate_ids']<0).any():raise ValueError('Integer candidate IDs required')
    if (np.diff(np.sort(x['candidate_ids'],axis=1),axis=1)==0).any():raise ValueError('Ten distinct candidates required')
    if x['mask'].shape!=(n,10) or x['mask'].dtype.kind!='b' or not x['mask'].all():raise ValueError('This frozen protocol requires ten valid candidates')
    for k in ('behavior_scores','jev_scores','jev_entropy','jev_confidence'):
        if x[k].shape!=(n,10,7) or not np.isfinite(x[k]).all() or (x[k]<0).any() or (x[k]>1).any():raise ValueError(f'Invalid {k}')
        x[k]=x[k].astype('float32',copy=False)
    if not np.allclose(x['jev_scores'].sum(1),1,rtol=0,atol=2e-5):raise ValueError('Jev choices must sum to one per goal/slate')
    p=x['jev_scores'].astype('float64');entropy=np.clip(-(p*np.log(np.maximum(p,1e-300))).sum(1)/np.log(10),0,1)
    if not np.allclose(x['jev_entropy'],entropy[:,None,:],rtol=0,atol=2e-5) or not np.allclose(x['jev_confidence'],1-entropy[:,None,:],rtol=0,atol=2e-5):raise ValueError('Jev confidence/entropy differs from choice probabilities')
    for key,shape,alternate in [('popularity',(n,10),(n,10,1)),('history_length',(n,),(n,1))]:
        if x[key].shape==alternate:x[key]=x[key].reshape(shape)
        if x[key].shape!=shape or not np.isfinite(x[key]).all() or (x[key]<0).any():raise ValueError(f'Invalid {key}')
        x[key]=x[key].astype('float32',copy=False)
    user_routes(x['user_ids'])
    return x


class Candidate:
    def __init__(self,bundle):
        self.bundle=Path(bundle).resolve();self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if sha(self.bundle/'manifest.json')!=(self.bundle/'manifest.sha256').read_text().split()[0]:raise ValueError('Manifest hash differs')
        for relative,h in self.manifest['files'].items():
            if sha(self.path(relative))!=h:raise ValueError(f'Bundle file changed: {relative}')
        if sha(Path(__file__))!=self.manifest['files']['candidate_runtime.py']:raise ValueError('Loaded inference runtime differs')
        package=Path(paretojev.__file__).resolve().parent
        source={p.relative_to(package).as_posix():sha(p) for p in sorted(package.rglob('*.py'))}
        if canonical(source)!=self.manifest['model_source_sha256']:raise ValueError('Loaded model/encoder source differs')
        for name in ('numpy','torch','threadpoolctl'):
            if importlib.metadata.version(name)!=self.manifest['runtime']['packages'][name]:raise ValueError(f'Install the recorded {name} version for this reproducible candidate')
        self.models={};self.specs={}
        for fold,row in self.manifest['routes'].items():
            self.specs[int(fold)]=json.loads(self.path(row['feature_spec']).read_text())
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

    def path(self,relative):
        p=(self.bundle/relative).resolve()
        if Path(relative).is_absolute() or not p.is_relative_to(self.bundle):raise ValueError('Bundle paths must be relative and contained')
        return p

    def model(self,spec):
        path=spec['path']
        if path not in self.models:
            if sha(self.path(path))!=spec['sha256']:raise ValueError('Checkpoint changed before model load')
            payload=torch.load(self.path(path),map_location='cpu',weights_only=True)
            model=EMERReranker(EMERConfig(**payload['model_config']));model.load_state_dict(payload['state']);model.eval()
            if model.config.use_jev!=spec['use_jev'] or sum(p.numel() for p in model.parameters())!=43345:raise ValueError('Model arm/capacity differs')
            self.models[path]=model
        return self.models[path]

    @torch.no_grad()
    def predict(self,values,arm='emer_jev',seed=7,batch_size=32):
        if arm not in ('emer','emer_jev','anchor') or seed not in self.manifest['seeds']:raise ValueError('Unknown arm/seed')
        if batch_size!=32:raise ValueError('Use registered batch_size32 for exact replay')
        x=validate_inputs(values);route=self.manifest['routing'];folds=user_routes(x['user_ids'],route['partition_seed'],route['folds'])
        n=len(folds);scores=np.zeros((n,10),dtype='float32');used=np.zeros(n,bool)
        with threadpool_limits(limits=2):
            for f in range(route['folds']):
                ids=np.flatnonzero(folds==f)
                if not len(ids):continue
                spec=self.manifest['routes'][str(f)]['seeds'][str(seed)][arm];model=self.model(spec)
                encoded=encode_features(x['raw_features'][ids],self.specs[f])
                for start in range(0,len(ids),batch_size):
                    selected=ids[start:start+batch_size];b=len(selected)
                    batch=SlateBatch(candidate_features=torch.from_numpy(encoded[start:start+b]),
                        behavior_scores=torch.from_numpy(x['behavior_scores'][selected]),jev_scores=torch.from_numpy(x['jev_scores'][selected]),
                        jev_entropy=torch.from_numpy(x['jev_entropy'][selected]),jev_confidence=torch.from_numpy(x['jev_confidence'][selected]),
                        labels=torch.zeros((b,10,7)),preference_weights=torch.full((b,7),1/7),mask=torch.from_numpy(x['mask'][selected]),
                        popularity=torch.from_numpy(x['popularity'][selected,:,None]),history_length=torch.from_numpy(x['history_length'][selected,None]))
                    scores[selected]=model(batch)['score_mean'].numpy()
                used[ids]=model.config.use_jev
        order=np.argsort(-scores,axis=1,kind='stable')
        return dict(scores=scores,ranking_indices=order,ranked_candidate_ids=np.take_along_axis(x['candidate_ids'],order,axis=1),fold_ids=folds,uses_jev=used)
