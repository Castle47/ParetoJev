#!/usr/bin/env python3
"""Prepare a full-history, all-user, explicitly target-sampled 27K pilot."""
import argparse
import fcntl
from pathlib import Path
import numpy as np
from paretojev.data.kuairand_27k import discover_data
from paretojev.data.kuairand_pilot import ingest, prepare_samples, user_snapshot
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--config', type=Path, default=Path('configs/kuairand_27k_behavior_pilot.json'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(args.config)
        data = discover_data(args.data_root)
        registration = args.output / 'registration.json'
        if registration.exists() and read_json(registration) != config:
            raise ValueError('Existing preparation has a different registration')
        atomic_write_json(registration, config)
        events = ingest(data, args.audit, args.output)
        samples = prepare_samples(args.output, config)
        with np.load(args.output / 'train.npz') as train:
            profile = user_snapshot(data, args.output, train['user'])
        atomic_write_json(args.output / 'preparation.json', dict(events=events, samples=samples,
            user_snapshot=profile, config_sha256=sha256_file(args.config),
            source_sha256=sha256_file(Path(__file__).parents[1] / 'src/paretojev/data/kuairand_pilot.py')))
        print('Preparation complete', flush=True)


if __name__ == '__main__': main()
