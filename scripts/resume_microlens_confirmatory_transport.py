"""Audited operational continuation after an inspected transient network failure.

Keeps the original research registration and source files immutable. A separate
amendment grants at most 20 extra *attempts*, never additional users or decisions.
No internal retries: each resumed run still stops on the first error.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

import requests

import run_microlens_confirmatory as core
from paretojev.jev.cache import DecisionCache
from paretojev.jev.durable_budget import DurableBudgetSession
from paretojev.jev.public_demo import PublicDemoJevClient, score_public_demo_noul
from paretojev.jev.prompts import select_objectives
from paretojev.jev.runner import run_score_jsonl, ScoreJsonlConfig
from paretojev.utils.io import read_json, read_jsonl, atomic_write_json
from paretojev.utils.provenance import sha256_file


def is_transient_transport(error):
    kind, message = error.get('error_type'), error.get('error', '')
    return kind in {'ReadTimeout', 'ConnectTimeout'} or (
        kind == 'SSLError' and 'UNEXPECTED_EOF_WHILE_READING' in message)


def amended_limit(registration, amendment):
    original = registration['cohort']['max_new_http_calls_lifetime']
    if amendment.get('original_max_attempts') != original or amendment.get('extra_transport_attempts') != 20:
        raise ValueError('Only the recorded 20-attempt operational contingency is allowed')
    return original + 20


def main():
    reg = core.verify(); core.validate_frozen_predictions()
    amendment_path = core.REPORT / 'transport_amendment.json'
    amendment = read_json(amendment_path)
    for name, digest in amendment['anchors'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'Operational amendment anchor changed: {name}')
    maximum = amended_limit(reg, amendment)
    with (core.ROOT / 'scoring.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = read_jsonl(core.ROOT / 'scores.jsonl')
        if len(completed) != core.N_USERS:
            latest = read_json(core.ROOT / 'scoring_report.json')
            errors = read_jsonl(core.ROOT / 'errors.jsonl')
            if latest['status'] != 'failed' or not errors or not is_transient_transport(errors[-1]):
                raise ValueError('Only an explicitly inspected transient transport failure can resume')
            health_url = reg['decision_service']['endpoint'].replace('/api/decide', '/api/health')
            health = requests.get(health_url, timeout=25); health.raise_for_status()
            body = health.json()
            if body.get('status') != 'ok' or body.get('model') != reg['decision_service']['model'] or body.get('configured') is not True:
                raise ValueError('Service health/model has not recovered')
        session = DurableBudgetSession(core.ROOT / 'http_attempts.jsonl', maximum)
        cache = DecisionCache(core.ROOT / 'requests.sqlite3')
        client = PublicDemoJevClient(revision=reg['decision_service']['expected_revision'], cache=cache,
            max_http_calls=max(1, maximum-session.used), minimum_interval=1.0)
        client.session.close(); client.session = session
        identity = {'registration_sha256': sha256_file(core.REPORT / 'registration.json'),
            'prediction_freeze_sha256': sha256_file(core.REPORT / 'prediction_freeze.json')}
        if read_json(core.ROOT / 'collection_identity.json') != identity:
            raise ValueError('Research or checkpoint identity changed')
        runs = core.ROOT / 'collection_runs'
        report = runs / f'{len(list(runs.glob("*.json")))+1:04d}.json'
        try:
            run_score_jsonl(core.DATA / 'test/jev_input.jsonl', core.ROOT / 'scores.jsonl',
                client=client, score_function=score_public_demo_noul, objectives=select_objectives(['click']),
                config=ScoreJsonlConfig(provider=client.provider, mode='noul', objective_names=('click',), durable_every=1),
                report_path=report, error_path=core.ROOT / 'errors.jsonl', cache_count=cache.count)
        finally:
            client.session.close()
            if report.exists():
                result = read_json(report)
                result.update(identity=identity, lifetime_attempts=session.used,
                    operational_amendment_sha256=sha256_file(amendment_path), maximum_attempts=maximum)
                atomic_write_json(report, result); atomic_write_json(core.ROOT / 'scoring_report.json', result)
                print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
