#!/usr/bin/env python3
"""Reuse the validated durable Choice scorer, recording HTTP wall latency too."""
import argparse
import time
from pathlib import Path

from paretojev.jev.durable_budget import DurableBudgetSession
from paretojev.utils.io import append_jsonl_row, read_json
from paretojev.utils.provenance import sha256_file, canonical_hash
from paretojev.experiments.comparison import _output_lock
import score_kuairand_core


class TimedSession(DurableBudgetSession):
    def post(self,url,**kwargs):
        start=time.perf_counter();error=None;response=None
        try:
            response=super().post(url,**kwargs)
            return response
        except Exception as exc:
            error=type(exc).__name__;raise
        finally:
            with (self.ledger_path.parent/'http_latency.jsonl').open('a') as handle:
                append_jsonl_row(handle,dict(attempt=self.used,payload_sha256=canonical_hash(kwargs.get('json')),
                    wall_seconds=time.perf_counter()-start,status=getattr(response,'status_code',None),error=error),flush=True,fsync=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/emer_jev.json'))
    a=p.parse_args();a.output=a.data/'jev';a.output.mkdir(parents=True,exist_ok=True)
    c=read_json(a.config);prep=read_json(a.data/'preparation.json')
    if prep['config_hash']!=canonical_hash(c) or prep['slates']*len(c['objectives'])!=c['jev']['planned_calls']:
        raise ValueError('Scoring cohort differs from registration')
    if not (a.data/'selection.json').exists():raise ValueError('Freeze both fitted arms before fresh scoring')
    score_kuairand_core.DurableBudgetSession=TimedSession
    with _output_lock(a.output):score_kuairand_core.score(a)
