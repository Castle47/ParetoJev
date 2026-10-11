#!/usr/bin/env python3
"""Independent process replay of registered pilot checkpoints; no fitting or HTTP."""
import argparse
from pathlib import Path
import numpy as np
import torch
from train_kuairand_behavior_pilot import predict, metrics
from paretojev.models.kuairand_pilot import KuaiRandPilotModel
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import canonical_hash, sha256_file


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = read_json(args.run / 'results.json'); config = result['config']
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if sha256_file(args.data / 'validation.npz') != result['sample_manifest']['splits']['validation']['sha256']:
        raise ValueError('Validation sample changed')
    with np.load(args.data / 'validation.npz') as z: val = {k: z[k] for k in z.files}
    checks = {}
    for regime in config['feature_regimes']:
        evidence = result['models'][regime]
        checkpoint = args.run / f'{regime}.pt'
        pred_path = args.run / f'{regime}-validation-predictions.npy'
        if sha256_file(checkpoint) != evidence['checkpoint_sha256']: raise ValueError('Checkpoint changed')
        if sha256_file(pred_path) != evidence['predictions_sha256']: raise ValueError('Saved prediction changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['config_hash'] != canonical_hash(config): raise ValueError('Registration changed')
        if saved['sample_manifest_sha256'] != sha256_file(args.data / 'samples.json'):
            raise ValueError('Sample manifest changed')
        profiles = None
        if regime == 'released_user_snapshot':
            spec = read_json(args.data / 'user_snapshot.json')
            if sha256_file(args.data / 'user_snapshot.npy') != spec['encoded_sha256']: raise ValueError('Profiles changed')
            profiles = np.load(args.data / 'user_snapshot.npy')
        model = KuaiRandPilotModel(config, profile_cardinalities=saved['profile_cardinalities'])
        model.load_state_dict(saved['state_dict'])
        actual = predict(model, val, profiles, 'cpu', config['batch_size'])
        prediction_equal = np.array_equal(actual, np.load(pred_path))
        metrics_equal = metrics(val['labels'], actual, val['weight']) == evidence['validation']
        if not prediction_equal or not metrics_equal: raise ValueError(f'Replay failed: {regime}')
        checks[regime] = dict(predictions_exact=prediction_equal, metrics_exact=metrics_equal,
                              rows=len(actual), checkpoint_sha256=sha256_file(checkpoint))
        print(f'{regime}: exact replay passed', flush=True)
    atomic_write_json(args.output, dict(status='passed', scope='fresh_process_same_machine_cpu',
        full_training_replayed=False, new_jev_calls=0, checks=checks))


if __name__ == '__main__': main()
