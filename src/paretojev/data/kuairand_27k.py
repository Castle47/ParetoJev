"""Bounded-memory acquisition inventory and schema/feedback audit for KuaiRand-27K.

This module does not train a model, subsample users, or treat released whole-month
statistics as pre-exposure features. CSV chunks are never assumed chronological.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from paretojev.utils.io import read_json

USER_COUNT = 27285
ITEM_COUNT = 32038725
LOG_GROUPS = {
    'early_standard': (
        'log_standard_4_08_to_4_21_27k_part1.csv',
        'log_standard_4_08_to_4_21_27k_part2.csv',
    ),
    'late_standard': (
        'log_standard_4_22_to_5_08_27k_part1.csv',
        'log_standard_4_22_to_5_08_27k_part2.csv',
    ),
    'random': ('log_random_4_22_to_5_08_27k.csv',),
}
USER_FILE = 'user_features_27k.csv'
ITEM_FILE = 'video_features_basic_27k.csv'
STATISTIC_FILES = tuple(f'video_features_statistic_27k_part{i}.csv' for i in (1, 2, 3))
CORE_FILES = tuple(n for names in LOG_GROUPS.values() for n in names) + (USER_FILE, ITEM_FILE) + STATISTIC_FILES
BINARY_LABELS = ('is_click', 'long_view', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate')
REQUIRED = ('user_id', 'video_id', 'date', 'time_ms', 'tab', 'is_rand', 'play_time_ms', 'duration_ms', *BINARY_LABELS)
USER_FIELDS = ('user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
    'follow_user_num', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
    'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range',
    *(f'onehot_feat{i}' for i in range(18)))
ITEM_FIELDS = ('author_id', 'video_type', 'upload_dt', 'upload_type', 'visible_status',
    'video_duration', 'server_width', 'server_height', 'music_id', 'music_type', 'tag')
ASSETS = (
    {'name': 'KuaiRand-27K.tar.gz', 'url': 'https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz',
     'md5': '3e3c799a24e2d23a4d2c757fbf9adf59', 'role': 'full_27k_core'},
    {'name': 'kuairand_video_captions.csv', 'url': 'https://zenodo.org/records/18159199/files/kuairand_video_captions.csv',
     'md5': '9fd4e7587ffd0c8dc289d013858fb229', 'role': 'content_supplement'},
    {'name': 'kuairand_video_categories.csv', 'url': 'https://zenodo.org/records/18159199/files/kuairand_video_categories.csv',
     'md5': '8a4c772d3d8f8cf65fa1e1eb896f4177', 'role': 'content_supplement'},
)


def storage_plan(root: str | Path, minimum_free_gib: float = 100) -> dict:
    root = Path(root).expanduser().resolve()
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    allocated = 0
    for asset in ASSETS:
        for suffix in ('', '.part'):
            path = root/'downloads'/(asset['name']+suffix)
            if path.is_file() and not path.is_symlink():
                allocated += path.stat().st_size
                break
    extraction = root/'extracted/extraction_manifest.json'
    if extraction.is_file():
        metadata = read_json(extraction)
        if metadata.get('archive_md5') == ASSETS[0]['md5']:
            for name, entry in metadata.get('files', {}).items():
                path = root/'extracted'/name
                if path.resolve().is_relative_to((root/'extracted').resolve()) and path.is_file() and not path.is_symlink() and path.stat().st_size == entry.get('bytes'):
                    allocated += path.stat().st_size
    remaining = max(2 * 2**30, int(minimum_free_gib * 2**30) - allocated)
    return {'data_root': str(root), 'filesystem_probe': str(ancestor), 'free_bytes': free,
        'free_gib': free / 2**30, 'initial_workspace_budget_gib': minimum_free_gib,
        'existing_dataset_bytes': allocated, 'required_remaining_free_gib': remaining / 2**30,
        'status': 'ready' if free >= remaining else 'blocked_storage',
        'planning_basis': 'Published core expands to about 46 GB; supplements about 6.9 GB. Reserve archive and preprocessing space. 100 GiB is a conservative workspace planning floor, not a published dataset size.'}


def discover_data(root: str | Path) -> Path:
    root = Path(root)
    candidates = (root, root/'data', root/'KuaiRand-27K/data', root/'extracted/KuaiRand-27K/data')
    found = [p for p in candidates if all((p/name).is_file() for name in CORE_FILES)]
    if len(found) != 1:
        raise ValueError('Need exactly one complete 27K directory containing all ten core CSVs; no Pure/1K fallback')
    return found[0]


def feature_contract() -> dict:
    return {
        'dataset': 'KuaiRand-27K', 'published_user_feature_count': len(USER_FIELDS),
        'published_item_feature_count': 62, 'item_basic_fields_expected': list(ITEM_FIELDS),
        'user_fields_expected': list(USER_FIELDS), 'raw_binary_labels': list(BINARY_LABELS),
        'derived_legacy_objective': {'deep_engagement': 'max(is_comment,is_forward), explicitly derived from real feedback'},
        'event_inputs': ['user_id', 'video_id', 'time_ms', 'tab'],
        'sequence': {'history_length': 50, 'history_end': 'strictly before current time_ms',
                     'source': 'all retained 27K user history; no ID-only item-pool filtering'},
        'user_snapshot_status': 'timing and mutability unverified; do not assume all 30 fields are pre-exposure',
        'item_basic_status': 'audit availability and upload time; metadata-as-released assumptions must be disclosed',
        'blocked_features': {'video_features_statistic_27k_part*.csv': 'whole-month aggregate; exclude from strict temporal model inputs',
            'current_row_feedback': 'labels only, never input features'},
        'historical_aggregates': 'recompute from strictly earlier log events; equal timestamps must not see one another',
        'is_click_semantics': 'click in two-column UI; valid_play in single-column UI. Map/stratify tab before calling this CTR.',
        'content': 'join caption/category by documented IDs and measure coverage; never fabricate missing captions',
        'approved_real_input_feature_count': None,
        'training_ready': False,
        'readiness_gates': ['full-data counts and hashes verified', 'label prevalence by tab/date audited',
            'temporal feature availability resolved', 'content ID alignment/coverage verified',
            'out-of-core chronology and history store prepared', 'memory-bounded rich-feature pXTR trainer validated'],
    }


class HashReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        self.eof = False

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count]); self.bytes_read += count
        else:
            self.eof = True
        return count


def audit_log(path: str | Path, *, random_log: bool, chunksize: int = 100_000,
              max_rows: int | None = None, user_count: int = USER_COUNT, item_count: int = ITEM_COUNT) -> dict:
    """Read one shard with O(chunk + catalog + users) memory, never concatenate."""
    if chunksize <= 0 or (max_rows is not None and max_rows <= 0):
        raise ValueError('Positive chunk/sample sizes required')
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(REQUIRED) - set(header)
    if missing:
        raise ValueError(f'{path.name} missing columns: {sorted(missing)}')
    user_counts = np.zeros(user_count, dtype=np.int64)
    seen_items = np.zeros(item_count, dtype=bool)
    positives = np.zeros(len(BINARY_LABELS), dtype=np.int64)
    total = invalid_duration = long_view_mismatch = time_backsteps = 0
    previous_time = minimum = maximum = None
    grouped: dict[tuple[int, int], dict] = {}
    before = path.stat()
    with path.open('rb') as raw:
        tracker = HashReader(raw)
        with io.BufferedReader(tracker) as buffered:
            for chunk in pd.read_csv(buffered, usecols=list(REQUIRED), dtype='int64', chunksize=chunksize, nrows=max_rows):
                users, items, times = (chunk[c].to_numpy() for c in ('user_id', 'video_id', 'time_ms'))
                if len(chunk) == 0:
                    continue
                if (users < 0).any() or (users >= user_count).any() or (items < 0).any() or (items >= item_count).any():
                    raise ValueError('ID outside declared 27K index space; investigate version/alignment')
                if not np.isin(chunk[list(BINARY_LABELS) + ['is_rand']].to_numpy(), (0, 1)).all():
                    raise ValueError('Nonbinary/missing outcome or is_rand; do not impute silently')
                if not chunk.is_rand.eq(int(random_log)).all():
                    raise ValueError('Random and standard tracks mixed')
                if not chunk.tab.between(0, 14).all():
                    raise ValueError('Unknown tab; scene definitions need review')
                np.add.at(user_counts, users, 1); seen_items[items] = True
                positives += chunk[list(BINARY_LABELS)].sum().to_numpy(dtype=np.int64)
                total += len(chunk)
                if total % (chunksize * 20) == 0:
                    print(f'Auditing {path.name}: {total} rows', flush=True)
                minimum = int(times.min()) if minimum is None else min(minimum, int(times.min()))
                maximum = int(times.max()) if maximum is None else max(maximum, int(times.max()))
                time_backsteps += int((np.diff(times) < 0).sum())
                if previous_time is not None:
                    time_backsteps += int(times[0] < previous_time)
                previous_time = int(times[-1])
                duration, play = chunk.duration_ms.to_numpy(), chunk.play_time_ms.to_numpy()
                known = (duration > 0) & (play >= 0)
                invalid_duration += int((~known).sum())
                expected = play >= np.minimum(duration, 18_000)
                long_view_mismatch += int((known & (expected != chunk.long_view.to_numpy())).sum())
                for (date, tab), frame in chunk.groupby(['date', 'tab'], sort=False):
                    entry = grouped.setdefault((int(date), int(tab)), {'rows': 0, **{name: 0 for name in BINARY_LABELS}})
                    entry['rows'] += len(frame)
                    for name in BINARY_LABELS:
                        entry[name] += int(frame[name].sum())
        digest = tracker.digest.hexdigest(); bytes_read = tracker.bytes_read
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while auditing')
    full = max_rows is None
    if full and bytes_read != before.st_size:
        raise ValueError('Full scan did not consume the complete source')
    if total == 0:
        raise ValueError('Empty log')
    return {'file': path.name, 'scope': 'full_file' if full else 'prefix_only', 'rows': total,
        'source_bytes': before.st_size, 'source_sha256': digest if full else None,
        'columns': header, 'distinct_users_observed': int((user_counts > 0).sum()),
        'distinct_items_observed': int(seen_items.sum()), 'min_time_ms': minimum, 'max_time_ms': maximum,
        'min_date': min(d for d, _ in grouped), 'max_date': max(d for d, _ in grouped),
        'rows_with_invalid_duration_or_play_time': invalid_duration,
        'long_view_rule_mismatches_on_valid_duration': long_view_mismatch,
        'global_timestamp_backsteps': time_backsteps,
        'must_not_assume_file_or_shard_order_is_chronological': True,
        'positive_counts': dict(zip(BINARY_LABELS, map(int, positives), strict=True)),
        'prevalence': dict(zip(BINARY_LABELS, (positives/total).tolist(), strict=True)),
        'by_date_and_tab': [{'date': d, 'tab': t, **v} for (d, t), v in sorted(grouped.items())],
        'user_event_counts': user_counts.tolist()}


def audit_directory(root: str | Path, *, chunksize: int = 100_000, max_rows_per_file: int | None = None) -> dict:
    directory = discover_data(root)
    result = {'dataset': 'KuaiRand-27K', 'scope': 'full_log_audit' if max_rows_per_file is None else 'prefix_only_audit',
        'data_directory': str(directory.resolve()), 'benchmark_claim': False,
        'training_ready': False, 'feature_contract': feature_contract(), 'logs': {}, 'metadata_schemas': {}}
    for group, names in LOG_GROUPS.items():
        reports = []
        for name in names:
            entry = audit_log(directory/name, random_log=group == 'random', chunksize=chunksize, max_rows=max_rows_per_file)
            reports.append(entry)
            print(f'Audited {name}: {entry["rows"]} rows ({entry["scope"]})', flush=True)
        result['logs'][group] = reports
        date_low, date_high = (20220408, 20220421) if group == 'early_standard' else (20220422, 20220508)
        if any(r['min_date'] < date_low or r['max_date'] > date_high for r in reports):
            raise ValueError(f'{group} contains dates outside the published interval')
    for name, expected in [(USER_FILE, ['user_id', *USER_FIELDS]), (ITEM_FILE, ['video_id', *ITEM_FIELDS])]:
        columns = pd.read_csv(directory/name, nrows=0).columns.tolist()
        result['metadata_schemas'][name] = {'observed_columns': columns, 'missing_expected_columns': sorted(set(expected)-set(columns)),
            'value_quality_audited': False, 'pre_exposure_availability_verified': False}
    for name in STATISTIC_FILES:
        result['metadata_schemas'][name] = {'observed_columns': pd.read_csv(directory/name,nrows=0).columns.tolist(),
            'eligible_model_input': False, 'reason': 'published whole-month aggregates'}
    if max_rows_per_file is None:
        standard = sum(x['rows'] for group in ('early_standard','late_standard') for x in result['logs'][group])
        random = sum(x['rows'] for x in result['logs']['random'])
        result['published_count_check'] = {'standard_rows': standard, 'expected_standard_rows': 322278385,
            'random_rows': random, 'expected_random_rows': 1186059,
            'matches': standard == 322278385 and random == 1186059}
    return result
