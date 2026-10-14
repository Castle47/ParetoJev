"""Replay the original stratified reservoir using source user partitions only."""
import heapq
from pathlib import Path
import numpy as np
from paretojev.data.kuairand_core import eligible_blocks,strata_for
from paretojev.utils.io import read_json,atomic_write_json
from paretojev.utils.provenance import sha256_file,canonical_hash
from prepare_emer_jev import DAY_EVENT


def audit_sampling(data,config):
    out=data/'sampling_replay.json';raw_path=data/'raw_slates.npz'
    identity=dict(config_hash=canonical_hash(config),raw_sha256=sha256_file(raw_path),
        partition_manifest_sha256=sha256_file(data/'audit-partitions-v1/manifest.json'),
        script_sha256=sha256_file(Path(__file__)))
    if out.exists():
        previous=read_json(out)
        if previous['identity']!=identity or not previous['exact_selection_replayed']:raise ValueError('Sampling audit identity differs')
        return previous
    partition=read_json(data/'audit-partitions-v1/manifest.json')
    if partition['source_sha256']!=read_json(data/'preparation.json')['sources']['sha256']:
        raise ValueError('Source partitions differ')
    heaps=[[] for _ in range(6)];pop=np.zeros(6,dtype='int64');users=0
    for filename,h in sorted(partition['files'].items()):
        path=data/'audit-partitions-v1'/filename
        if sha256_file(path)!=h:raise ValueError('Source partition changed')
        part=np.fromfile(path,dtype=DAY_EVENT);part=part[np.argsort(part['user'],kind='stable')]
        starts=np.r_[0,np.flatnonzero(np.diff(part['user']))+1,len(part)]
        for left,right in zip(starts[:-1],starts[1:]):
            events=part[left:right];user=int(events['user'][0]);events=events[np.argsort(events['time'],kind='stable')]
            blocks=eligible_blocks(events,config['fresh_test_start_ms'],config['fresh_test_end_ms'],config['slate_size'],config['maximum_slate_span_ms'])
            strata=strata_for(events,blocks,config['stratum_priority_bits'])
            priorities=np.random.default_rng(np.random.SeedSequence([config['sample_seed'],user])).random(len(blocks))
            for block,stratum,priority in zip(blocks,strata,priorities):
                pop[stratum]+=1;heap=heaps[stratum];limit=config['test_stratum_budgets'][stratum]
                records=tuple(int(v) for v in events['record'][block])
                entry=(float(priority),records[0],user,records,int(events['time'][block[0]]))
                if len(heap)<limit:heapq.heappush(heap,entry)
                elif priority>heap[0][0]:heapq.heapreplace(heap,entry)
            users+=1
        if users%5000<512:print('Sampling replay users',users,flush=True)
    selected=[(int(stratum),entry) for stratum,heap in enumerate(heaps) for entry in heap]
    selected.sort(key=lambda row:(row[1][4],row[1][2]))
    with np.load(raw_path) as z:raw={k:z[k] for k in ('records','user','stratum','weight')}
    if len(selected)!=len(raw['user']):raise ValueError('Sampling count differs')
    if pop.tolist()!=read_json(data/'preparation.json')['populations']:raise ValueError('Eligible populations differ')
    for i,(stratum,entry) in enumerate(selected):
        user=entry[2];permutation=np.random.default_rng(np.random.SeedSequence([config['sample_seed'],user,i])).permutation(config['slate_size'])
        expected=np.array(entry[3])[permutation]
        if raw['user'][i]!=user or raw['stratum'][i]!=stratum or not np.array_equal(raw['records'][i],expected):
            raise ValueError('Frozen random sample/order does not replay')
        if raw['weight'][i]!=pop[stratum]/config['test_stratum_budgets'][stratum]:raise ValueError('Inclusion weight differs')
    result=dict(identity=identity,exact_selection_replayed=True,populations=pop.tolist(),slates=len(selected),
        all_source_users_checked=users,performance_evaluated=False)
    atomic_write_json(out,result);return result
