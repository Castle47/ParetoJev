"""Partial-score and provenance guards before new-user outcome inspection."""
import importlib.util
from pathlib import Path
import sys

import pytest

from paretojev.utils.io import atomic_write_json


def load_script(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('confirmatory_test_module', scripts / 'run_microlens_confirmatory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_or_duplicate_cohort_cannot_be_evaluated(tmp_path, monkeypatch):
    run = load_script(monkeypatch)
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'verify', lambda: {'cohort': {'users': list(range(700))}})
    monkeypatch.setattr(run, 'validate_frozen_predictions', lambda: None)
    monkeypatch.setattr(run, 'read_jsonl', lambda path: [{'slate_id': i} for i in range(699)])
    with pytest.raises(ValueError, match='No partial-cohort evaluation'):
        run.require_complete()
    monkeypatch.setattr(run, 'read_jsonl', lambda path: [{'slate_id': 0}] * 700)
    with pytest.raises(ValueError, match='No partial-cohort evaluation'):
        run.require_complete()


def test_changed_source_cannot_resume_experiment(tmp_path, monkeypatch):
    run = load_script(monkeypatch)
    monkeypatch.setattr(run, 'REPORT', tmp_path)
    source = tmp_path / 'source.py'; source.write_text('changed')
    atomic_write_json(tmp_path / 'registration.json', {'input_hashes': {},
        'prepared_hashes': {}, 'source_hashes': {str(source): 'wrong digest'}})
    with pytest.raises(ValueError, match='Frozen experiment file changed'):
        run.verify()
