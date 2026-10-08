import copy

import numpy as np
import pytest

from paretojev.data.microlens import _within_slate_rank_normalize
from paretojev.jev.cache import DecisionCache
from paretojev.jev.client import DecisionAPIError
from paretojev.jev.public_demo import PublicDemoJevClient, validate_public_response

REVISION = "7ce153d6f7933769527939bcb92ec3f3d8292884"


def response():
    return {"kind": "noul", "options": ["false", "true"], "probabilities": [0.2, 0.8],
            "choice_index": 1, "choice": "true", "model": "autotrust/JEV-27B",
            "revision": REVISION, "image_used": False, "elapsed_seconds": 0.12}


@pytest.mark.parametrize("changes", [
    {"model": "qwen"}, {"revision": "unavailable"}, {"kind": "choice"},
    {"options": ["true", "false"]}, {"probabilities": [float("nan"), 0.8]},
    {"probabilities": [0.3, 0.8]}, {"probabilities": [False, 1]},
    {"choice_index": 0}, {"choice": "false"}, {"image_used": True},
    {"elapsed_seconds": -1},
])
def test_rejects_wrong_model_revision_or_invalid_distribution(changes):
    data = response(); data.update(changes)
    with pytest.raises(DecisionAPIError):
        validate_public_response(data, REVISION)


def test_public_cache_preserves_exact_validated_response_and_limits_requests(tmp_path, monkeypatch):
    client = PublicDemoJevClient(revision=REVISION, cache=DecisionCache(tmp_path / "cache.sqlite3"), max_http_calls=1)
    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return copy.deepcopy(response())
    def post(url, **kwargs):
        calls.append((url, kwargs)); return Response()
    monkeypatch.setattr(client.session, "post", post)
    first = client.decide("noul", {"history": ["camera tutorial"]}, "Would this user watch A?")
    again = client.decide("noul", {"history": ["camera tutorial"]}, "Would this user watch A?")
    assert first == again == response()
    assert len(calls) == 1 and client.stats()["cache_hits"] == 1
    with pytest.raises(DecisionAPIError, match="budget"):
        client.decide("noul", {}, "A different request")


def test_client_rejects_oversized_prompts_without_network(tmp_path):
    client = PublicDemoJevClient(revision=REVISION, cache=DecisionCache(tmp_path / "cache.sqlite3"))
    with pytest.raises(ValueError, match="6000"):
        client.decide("noul", "x" * 6001, "q")
    with pytest.raises(ValueError, match="1000"):
        client.decide("noul", {}, "q" * 1001)
    assert client.stats()["http_calls"] == 0


def test_hybrid_rank_normalization_does_not_invent_information_from_ties():
    values = np.array([[0., 0., 0., 0.], [5., 1., 1., 9.]])
    result = _within_slate_rank_normalize(values)
    assert np.allclose(result[0], 0.5)
    assert result[1, 1] == result[1, 2]
    permutation = np.array([2, 0, 3, 1])
    assert np.allclose(_within_slate_rank_normalize(values[:, permutation]), result[:, permutation])
