"""Acquire and audit full KuaiRand-27K; never silently switch variants or sample users."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

from paretojev.data.acquisition import download_asset, extract_27k
from paretojev.data.kuairand_27k import ASSETS, CORE_FILES, audit_directory, feature_contract, storage_plan
from paretojev.utils.io import atomic_write_json, read_json
from paretojev.utils.provenance import runtime_provenance, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','download','extract','audit'])
    parser.add_argument('--root',type=Path,default=Path('data/raw/kuairand-27k'))
    parser.add_argument('--report',type=Path,default=Path('reports/kuairand-27k-readiness-2026-10-09/acquisition_plan.json'))
    parser.add_argument('--asset', action='append', choices=[a['name'] for a in ASSETS], help='Download only explicitly selected assets; other files remain unchanged')
    parser.add_argument('--range-mib',type=int,default=None,help='Bound each download request; final full checksum remains required')
    parser.add_argument('--transport-retries',type=int,default=0)
    parser.add_argument('--chunksize',type=int,default=100_000)
    parser.add_argument('--max-rows-per-file',type=int,default=None,help='Explicit prefix audit, never a full-dataset claim')
    args = parser.parse_args(); root = args.root.expanduser().resolve()
    plan = {'dataset':'KuaiRand-27K','storage':storage_plan(root),'sources':list(ASSETS),
        'expected_core_files':list(CORE_FILES),'feature_contract':feature_contract(),
        'full_core_dataset_acquired':False,'model_training_completed':False,
        'source_hashes':{str(path):sha256_file(path) for path in (Path(__file__).relative_to(Path.cwd()),
            Path('src/paretojev/data/acquisition.py'),Path('src/paretojev/data/kuairand_27k.py'),Path('configs/kuairand.yaml'))},
        'provenance':runtime_provenance()}
    if args.stage == 'plan':
        atomic_write_json(args.report,plan)
        print(f"27K plan: {plan['storage']['status']}; {plan['storage']['free_gib']:.2f} GiB free; report={args.report}")
        return
    if args.stage == 'download' and plan['storage']['status'] != 'ready':
        atomic_write_json(args.report,plan)
        raise OSError('Full 27K acquisition needs the registered 100 GiB free-space floor; provide a larger data directory. No download started.')
    root.mkdir(parents=True,exist_ok=True)
    # Verified final downloads are immutable. Extraction can read the completed
    # core while a different supplementary asset is still downloading.
    lock_name = {'download':'acquisition.lock','extract':'extraction.lock','audit':'audit.lock'}[args.stage]
    with (root/lock_name).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage == 'download':
            manifest_path = root/'download_manifest.json'
            manifest = read_json(manifest_path) if manifest_path.exists() else {'dataset':'KuaiRand-27K','assets':{}}
            for asset in ASSETS:
                if args.asset and asset['name'] not in args.asset:
                    continue
                result = download_asset(asset,root/'downloads',
                    request_bytes=args.range_mib * 2**20 if args.range_mib is not None else None,
                    max_transport_retries=args.transport_retries)
                manifest['assets'][asset['name']] = result
                atomic_write_json(manifest_path,manifest)
                print(f"Verified {asset['name']}: {result['bytes']} bytes",flush=True)
            manifest['complete'] = all(a['name'] in manifest['assets'] and
                manifest['assets'][a['name']]['md5'] == a['md5'] and
                (root/'downloads'/a['name']).is_file() and
                (root/'downloads'/a['name']).stat().st_size == manifest['assets'][a['name']]['bytes'] for a in ASSETS)
            atomic_write_json(manifest_path,manifest)
        elif args.stage == 'extract':
            extract_27k(root/'downloads/KuaiRand-27K.tar.gz',root/'extracted',expected_md5=ASSETS[0]['md5'])
        else:
            extracted = root/'extracted/extraction_manifest.json'
            if extracted.exists() and read_json(extracted).get('status') != 'complete':
                raise ValueError('Wait for complete atomic extraction before full audit')
            report = audit_directory(root,chunksize=args.chunksize,max_rows_per_file=args.max_rows_per_file)
            report['provenance'] = runtime_provenance()
            atomic_write_json(args.report,report)
            print(f"27K audit saved: {args.report}; scope={report['scope']}; training_ready=False")


if __name__ == '__main__':
    main()
