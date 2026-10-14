#!/usr/bin/env python3
"""Frozen independent-cohort Jev collection; exact development producer required."""
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import urllib.error
import urllib.request

MODEL='autotrust/JEV-27B'
REVISION='14b425269ed24452e0a5ad5a8f3e51d139647a88'
PROTOCOL='jev27-bare-v1'


def require_same_producer(identity, development):
    if identity!=development['source']:
        raise ValueError('Inference producer differs from frozen development; do not score test')


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def validate(response,options):
    if response.get('model')!=MODEL or response.get('protocol')!=PROTOCOL:
        raise ValueError('Unexpected native model/protocol')
    if response.get('kind')!='choice' or response.get('adaptation')!='native' or response.get('options')!=options:
        raise ValueError('Unexpected primitive, adaptation or candidate order')
    p=response.get('probabilities')
    if not isinstance(p,list) or len(p)!=len(options) or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in p):
        raise ValueError('Invalid probabilities')
    if abs(sum(p)-1)>1e-5:raise ValueError('Choice mass differs from one')
    index=response.get('choice_index')
    if type(index)!=int or not 0<=index<len(options) or p[index]!=max(p) or response.get('choice')!=options[index]:
        raise ValueError('Inconsistent winner')
    if response.get('implementation')=='pytorch-peft-group-v3' and 'group_shared_forwards' in response:
        if 'num_model_requests' in response or response['group_shared_forwards']!=2 or type(response.get('question_index'))!=int or not 0<=response['question_index']<7:
            raise ValueError('Invalid shared forward accounting')
    elif response.get('num_model_requests')!=1:raise ValueError('Unexpected System 1 forward count')
    if not isinstance(response.get('elapsed_seconds'),(int,float)) or not math.isfinite(response['elapsed_seconds']) or response['elapsed_seconds']<0:
        raise ValueError('Invalid elapsed time')
    return p


def post(endpoint,payload):
    req=urllib.request.Request(endpoint,canonical(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)


def load_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    for i,row in enumerate(rows):
        if row['slate_id']!=i or len(row['questions'])!=7 or len(row['options'])!=10 or len(row['candidate_ids'])!=10:
            raise ValueError('Invalid registered ten-candidate/seven-goal input')
        if row['split']!='test':raise ValueError('Confirmation packet must contain only test rows')
    return rows


def payload(row,j):
    return dict(kind='choice',state=row['state'],question=row['questions'][j],options=row['options'],thinking='off')


def group_payload(row):
    return dict(kind='choice',state=row['state'],questions=row['questions'],options=row['options'],thinking='off')


def validate_group(r,row):
    if r.get('model')!=MODEL or r.get('protocol')!=PROTOCOL or r.get('implementation')!='pytorch-peft-group-v3':
        raise ValueError('Unexpected grouped producer')
    if r.get('model_forward_calls')!=2 or r.get('question_count')!=7 or len(r.get('results',[]))!=7:
        raise ValueError('Invalid grouped coverage or forward accounting')
    for j,value in enumerate(r['results']):
        if value.get('question_index')!=j or value.get('implementation')!=r['implementation']:raise ValueError('Goal order or implementation differs')
        validate(value,row['options'])
    return r['results']


def producer(model_dir,endpoint):
    with urllib.request.urlopen(endpoint.rstrip('/')+'/info',timeout=30) as r:info=json.load(r)
    if info.get('implementation')!='pytorch-peft-group-v3' or info.get('huggingface_revision')!=REVISION:
        raise ValueError('Expected pinned grouped service before opening a new collection')
    files=sorted(set(model_dir.glob('model*.safetensors'))|set((model_dir/'adapter_vllm').glob('*'))|set((model_dir/'adapter').glob('*'))|
                 {model_dir/name for name in ('config.json','calibration.json','tokenizer.json','tokenizer_config.json','serve_decide.py','head.safetensors','judge_config.json')})
    if len(list(model_dir.glob('model-*.safetensors')))!=13:raise ValueError('Missing model shards')
    identities={str(p.relative_to(model_dir)):digest(p) for p in files if p.is_file()}
    packages={}
    for p in ('vllm','torch','transformers','huggingface_hub','peft','accelerate','flash-linear-attention','fla-core','causal-conv1d','triton'):
        try:packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:packages[p]=None
    if info.get('implementation','').startswith('pytorch-peft-'):
        identities['custom/torch_head_server.py']=digest('/root/torch_head_server.py')
    result=dict(model=MODEL,huggingface_revision=REVISION,files=identities,service_info=info,
        packages=packages,
        source='self-hosted native System 1; not public demo revision',endpoint=endpoint)
    result['producer_id']=hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def probe(rows,endpoint,out):
    sample=rows[:16];runs=[]
    for workers in (1,8,16):
        start=time.monotonic()
        def one(row):
            results=[]
            if workers==1:
                for j in range(7):
                    t=time.monotonic();r=post(endpoint,payload(row,j));validate(r,row['options'])
                    results.append(dict(slate_id=row['slate_id'],objective=j,response=r,wall_seconds=time.monotonic()-t))
            else:
                t=time.monotonic();r=post(endpoint+'_many',group_payload(row));children=validate_group(r,row)
                results=[dict(slate_id=row['slate_id'],objective=j,response=value,group_response=r,
                    wall_seconds=time.monotonic()-t) for j,value in enumerate(children)]
            return results
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            values=[r for rs in pool.map(one,sample) for r in rs]
        elapsed=time.monotonic()-start
        runs.append(dict(workers=workers,decisions=len(values),http_requests=len(values) if workers==1 else len(sample),
            elapsed_seconds=elapsed,qps=len(values)/elapsed,rows=values))
        save(out/f'probe-{workers}.json',runs[-1])
        print('probe',workers,'qps',runs[-1]['qps'],flush=True)
    base=runs[0]['rows'];maxdiff=max(abs(a-b) for run in runs[1:] for r0,r in zip(base,run['rows'])
        for a,b in zip(r0['response']['probabilities'],r['response']['probabilities']))
    report=dict(real_requests=sum(r['http_requests'] for r in runs),real_decisions=sum(r['decisions'] for r in runs),runs=[{k:v for k,v in r.items() if k!='rows'} for r in runs],
        maximum_probability_difference=maxdiff,stability_threshold=.01,passed=maxdiff<=.01,
        interpretation='Uncached single-question reference vs grouped shared-prefix inference; numerical and throughput check only, no recommendation-effect test.')
    save(out/'probe.json',report)
    if not report['passed']:raise ValueError('Native concurrent inference failed registered numerical stability check')


class Ledger:
    def __init__(self,path,max_attempts,deadline):
        self.lock=threading.Lock();self.max_attempts=max_attempts;self.deadline=deadline
        self.db=sqlite3.connect(path,check_same_thread=False);self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, row_id INT, goal INT, time REAL);'
            'CREATE TABLE IF NOT EXISTS results(row_id INT, goal INT, payload_hash TEXT, raw_json TEXT, wall REAL, PRIMARY KEY(row_id,goal));'
            'CREATE TABLE IF NOT EXISTS groups(row_id INT PRIMARY KEY,payload_hash TEXT,raw_json TEXT,wall REAL);')
    def cached(self,i,j,key):
        with self.lock:r=self.db.execute('SELECT payload_hash,raw_json FROM results WHERE row_id=? AND goal=?',(i,j)).fetchone()
        if r and r[0]!=key:raise ValueError('Cached input changed')
        return json.loads(r[1]) if r else None
    def reserve(self,i,j):
        with self.lock:
            if time.time()>=self.deadline:raise TimeoutError('Registered collection deadline reached; cache retained')
            count=self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
            if count>=self.max_attempts:raise RuntimeError('Lifetime request budget exhausted')
            self.db.execute('INSERT INTO attempts(row_id,goal,time) VALUES(?,?,?)',(i,j,time.time()));self.db.commit()
    def put(self,i,j,key,r,elapsed):
        with self.lock:
            self.db.execute('INSERT INTO results VALUES(?,?,?,?,?)',(i,j,key,canonical(r),elapsed));self.db.commit()
    def put_group(self,row,r,elapsed):
        validate_group(r,row)
        with self.lock:
            with self.db:
                self.db.execute('INSERT INTO groups VALUES(?,?,?,?)',(row['slate_id'],
                    hashlib.sha256(canonical(group_payload(row)).encode()).hexdigest(),canonical(r),elapsed))
                for j,child in enumerate(r['results']):
                    key=hashlib.sha256(canonical(payload(row,j)).encode()).hexdigest()
                    self.db.execute('INSERT INTO results VALUES(?,?,?,?,?)',(row['slate_id'],j,key,canonical(child),elapsed))


def collect(rows,a,out):
    end=dt.datetime.fromisoformat(a.stop_at)
    if end.tzinfo is None:raise ValueError('Explicit timezone required for collection deadline')
    ledger=Ledger(out/'native.sqlite3',a.max_attempts,end.timestamp());start=time.monotonic()
    def one(row):
        hits=[]
        for j in range(7):
            p=payload(row,j);key=hashlib.sha256(canonical(p).encode()).hexdigest();cached=ledger.cached(row['slate_id'],j,key)
            hits.append(cached is not None)
            if cached is not None:validate(cached,row['options'])
        if all(hits):return row['slate_id']
        if any(hits):raise ValueError('Partial grouped transaction; audit before resuming')
        for attempt in range(3):
            ledger.reserve(row['slate_id'],-1);t=time.monotonic()
            try:
                r=post(a.endpoint+'_many',group_payload(row));validate_group(r,row)
                ledger.put_group(row,r,time.monotonic()-t);break
            except urllib.error.HTTPError as e:
                if e.code not in (408,429,500,502,503,504) or attempt==2:raise
            except (urllib.error.URLError,TimeoutError):
                if attempt==2:raise
            time.sleep(2**attempt)
        return row['slate_id']
    # One HTTP group contains seven independent decisions sharing a token prefix.
    # Keep only 2*workers rows in flight; do not allocate 252K futures.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Deterministic scheduling only, without looking at scores or labels.
        # Keeping validation first preserves useful coverage if the rental ends.
        it=iter(sorted(rows,key=lambda r:(r['split']!='validation',r['slate_id'])));pending=set()
        for _ in range(a.workers*2):
            row=next(it,None)
            if row is not None:pending.add(pool.submit(one,row))
        done_count=0
        while pending:
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result();done_count+=1;row=next(it,None)
                if row is not None:pending.add(pool.submit(one,row))
            if done_count%50==0 or not pending:
                with ledger.lock:valid=ledger.db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
                progress=dict(validated_decisions=valid,planned_decisions=len(rows)*7,elapsed_seconds=time.monotonic()-start,complete=False)
                save(out/'progress.json',progress);print(progress,flush=True)
    import numpy as np
    p=np.zeros((len(rows),10,7))
    for row in rows:
        for j in range(7):
            key=hashlib.sha256(canonical(payload(row,j)).encode()).hexdigest()
            p[row['slate_id'],:,j]=validate(ledger.cached(row['slate_id'],j,key),row['options'])
    entropy=np.clip(-(p*np.log(np.maximum(p,1e-300))).sum(1)/math.log(10),0,1)
    entropy=np.broadcast_to(entropy[:,None,:],p.shape).copy()
    np.savez_compressed(out/'scores.npz',probabilities=p,entropy=entropy,confidence=1-entropy,
        candidate_ids=np.array([r['candidate_ids'] for r in rows]),slate_ids=np.arange(len(rows)),
        objective_names=np.array(rows[0]['objective_names']),model=np.array(MODEL),revision=np.array(REVISION))
    groups=ledger.db.execute('SELECT COUNT(*) FROM groups').fetchone()[0]
    attempts=ledger.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    if groups!=len(rows):raise ValueError('Missing original HTTP envelopes')
    ledger.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');ledger.db.close()
    save(out/'complete.json',dict(complete=True,valid_decisions=len(rows)*7,scores_sha256=digest(out/'scores.npz'),
        http_groups=groups,http_attempts=attempts,registration_sha256=digest(out/'registration.json'),public_cache_mixed=False))


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads(a.manifest.read_text())
        if digest(a.input)!=manifest['compressed_sha256']:raise ValueError('Uploaded compressed input differs from local manifest')
        with gzip.open(a.input,'rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=manifest['source_input_sha256']:
                raise ValueError('Uploaded raw input differs from audited development prompts')
        end=dt.datetime.fromisoformat(a.stop_at)
        if end.tzinfo is None or time.time()>=end.timestamp():raise ValueError('Missing timezone or expired collection deadline')
        rows=load_rows(a.input)
        if len(rows)!=50000 or len(rows)!=manifest['slates'] or not manifest['independent_test_included'] or manifest['labels_included']:
            raise ValueError('Frozen confirmation transfer scope changed')
        source=json.loads(a.source.read_text())
        if source['huggingface_revision']!=REVISION:raise ValueError('Pinned source revision differs')
        identity=producer(a.model_dir,a.endpoint)
        development=json.loads(a.development_registration.read_text())
        require_same_producer(identity,development)
        if manifest['development_registration_sha256']!=digest(a.development_registration):
            raise ValueError('Expected development producer receipt differs')
        registration=dict(input_sha256=digest(a.input),source=identity,slates=len(rows),goals=rows[0]['objective_names'],
            script_sha256=digest(__file__),max_attempts=a.max_attempts,stop_at=a.stop_at,workers=a.workers,
            separate_probe_http_requests=144,separate_probe_decisions=336,transfer_manifest_sha256=digest(a.manifest),
            transport='One real HTTP group per slate; groups table retains exact wire responses, results table retains aligned child decisions.',
            collection_order='ascending original test slate IDs; no adaptive sample replacement',
            kind='frozen independent test scores; no imported public responses',test_opened=True,
            development_registration_sha256=digest(a.development_registration),confirmation_config_sha256=manifest['confirmation_config_sha256'])
        reg=a.output/'registration.json'
        if reg.exists() and json.loads(reg.read_text())!=registration:raise ValueError('Native registration changed')
        save(reg,registration)
        if not (a.output/'probe.json').exists():probe(rows,a.endpoint,a.output)
        if not json.loads((a.output/'probe.json').read_text())['passed']:raise ValueError('Probe has not passed')
        if a.stage=='score':collect(rows,a,a.output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=Path('/root/autodl-tmp/JEV-27B'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/jev-service/source.json'))
    p.add_argument('--endpoint',default='http://127.0.0.1:8000/v1/decide');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--stop-at',required=True);p.add_argument('--max-attempts',type=int,default=55000)
    p.add_argument('--development-registration',type=Path,required=True)
    p.add_argument('--stage',choices=['probe','score'],default='probe');main(p.parse_args())
