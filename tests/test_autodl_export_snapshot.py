import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile

import pytest


@pytest.mark.parametrize("with_marker", [False, True])
def test_export_uses_registered_group_count_and_requires_completion_marker(tmp_path, with_marker):
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "registration.json").write_text(json.dumps({"slates": 3, "goals": ["a", "b"]}))
    with sqlite3.connect(output / "native.sqlite3") as db:
        db.executescript("CREATE TABLE groups(id INT); CREATE TABLE results(id INT);")
        db.executemany("INSERT INTO groups VALUES(?)", [(i,) for i in range(3)])
        db.executemany("INSERT INTO results VALUES(?)", [(i,) for i in range(6)])
    if with_marker:
        (output / "complete.json").write_text(json.dumps({"complete": True, "http_groups": 3, "valid_decisions": 6}))
    script = Path(__file__).resolve().parents[1] / "deployment/autodl/export_confirmation.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(output), "--archive-dir", str(tmp_path)],
                            check=True, capture_output=True, text=True)
    archive = json.loads(result.stdout)["archive"]
    with tarfile.open(archive) as tar:
        snapshot = json.load(tar.extractfile("jev-confirmation-v1/export_snapshot.json"))
    assert snapshot["expected_groups"] == 3
    assert snapshot["expected_decisions"] == 6
    assert snapshot["complete"] is with_marker
