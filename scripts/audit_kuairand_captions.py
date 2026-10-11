"""Stream a verified 27K caption table; report ID/text coverage, not recommendation metrics."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

from paretojev.data.kuairand_27k import HashReader, ITEM_COUNT
from paretojev.utils.io import read_json, atomic_write_json
from paretojev.utils.provenance import runtime_provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.root/'downloads/kuairand_video_captions.csv'
    published=read_json(args.root/'download_manifest.json')['assets'][source.name]
    seen=np.zeros(ITEM_COUNT,dtype=bool);available=np.zeros_like(seen)
    rows=duplicates=caption_nonempty=cover_nonempty=0
    before=source.stat()
    with source.open('rb') as raw:
        tracker=HashReader(raw)
        with io.BufferedReader(tracker) as buffered:
            reader=pd.read_csv(buffered,usecols=['final_video_id','caption','show_cover_text'],
                dtype={'final_video_id':'int64','caption':'string','show_cover_text':'string'},
                keep_default_na=False,chunksize=100_000)
            for frame in reader:
                ids=frame.final_video_id.to_numpy()
                if (ids<0).any() or (ids>=ITEM_COUNT).any():
                    raise ValueError('Caption IDs outside published 27K space; verify mapping before use')
                unique=np.unique(ids)
                duplicates+=len(ids)-len(unique)+int(seen[unique].sum());seen[unique]=True
                caption=frame.caption.str.strip().ne('').to_numpy(dtype=bool)
                cover=frame.show_cover_text.str.strip().ne('').to_numpy(dtype=bool)
                available[ids[caption|cover]]=True
                caption_nonempty+=int(caption.sum());cover_nonempty+=int(cover.sum());rows+=len(frame)
                if rows%2_000_000==0:print(f'Caption rows checked: {rows}',flush=True)
    if source.stat().st_size!=before.st_size or source.stat().st_mtime_ns!=before.st_mtime_ns:
        raise ValueError('Caption file changed during scan')
    if tracker.bytes_read!=before.st_size or tracker.digest.hexdigest()!=published['sha256']:
        raise ValueError('Scanned caption bytes differ from the verified download')
    atomic_write_json(args.output,{'dataset':'KuaiRand-27K','scope':'full_caption_inventory',
        'rows':rows,'distinct_video_ids':int(seen.sum()),'duplicate_rows':duplicates,
        'caption_nonempty_rows':caption_nonempty,'cover_nonempty_rows':cover_nonempty,
        'videos_with_either_text':int(available.sum()),'published_catalog_size':ITEM_COUNT,
        'text_coverage_fraction_of_published_catalog':float(available.sum()/ITEM_COUNT),
        'missingness_rule':'blank/whitespace only; literal NA/NULL strings are preserved',
        'source_sha256':tracker.digest.hexdigest(),'benchmark_claim':False,
        'observed_log_join_coverage_verified':False,'provenance':runtime_provenance()})
    print(f'Full caption inventory saved: {args.output}',flush=True)
