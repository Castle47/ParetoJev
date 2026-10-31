import datetime as dt
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery():
    path = Path(__file__).resolve().parents[1] / "deployment/autodl/resume_pilot_shard.py"
    spec = importlib.util.spec_from_file_location("pilot_shard_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resumes_partial_ranges_and_verifies_whole_file(recovery, monkeypatch, tmp_path):
    payload = bytes(range(128))
    monkeypatch.setattr(recovery, "SIZE", len(payload))
    monkeypatch.setattr(recovery, "PART_SIZE", len(payload) // 8)
    monkeypatch.setattr(recovery, "SHA256", hashlib.sha256(payload).hexdigest())
    requests = []

    def response(start, end):
        requests.append((start, end))
        return io.BytesIO(payload[start:end + 1])

    monkeypatch.setattr(recovery, "request_range", response)
    parts, model = tmp_path / "parts", tmp_path / "model"
    parts.mkdir()
    model.mkdir()
    for index in range(8):
        (parts / f"part{index}").write_bytes(payload[index * 16:index * 16 + 5])
    args = SimpleNamespace(parts=parts, model_dir=model,
                           stop_at=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=1)).isoformat())
    recovery.main(args)
    assert (model / recovery.NAME).read_bytes() == payload
    assert len(requests) == 16
    assert all((index * 16 + 5, index * 16 + 15) in requests for index in range(8))


def test_wrong_retained_prefix_is_not_appended(recovery, monkeypatch, tmp_path):
    (tmp_path / "part0").write_bytes(b"wrong")
    monkeypatch.setattr(recovery, "request_range", lambda start, end: io.BytesIO(b"right"))
    with pytest.raises(ValueError, match="wrong origin"):
        recovery.resume_part(0, tmp_path, recovery.time.time() + 60)
    assert (tmp_path / "part0").read_bytes() == b"wrong"


def test_unhonored_http_range_is_closed_and_rejected(recovery, monkeypatch):
    response = io.BytesIO(b"whole file")
    response.status = 200
    response.headers = {}
    monkeypatch.setattr(recovery.urllib.request, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="exact requested range"):
        recovery.request_range(4, 7)
    assert response.closed
