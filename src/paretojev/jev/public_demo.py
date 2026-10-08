"""Bounded, revision-checked access to the author's public text decision demo.

This is an AutoTrust JEV-27B deployment, not TypeSafe's hosted Jev and not the
native VL checkpoint. Use small pilots, one request at a time, and stop on errors.
Public source: huggingface.co/spaces/autotrust/JEV-27B-Demo (app.py).
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Iterable

import numpy as np
import requests

from paretojev.jev.cache import DecisionCache
from paretojev.jev.client import DecisionAPIError
from paretojev.jev.prompts import DecisionObjective, build_open_jev_noul_requests
from paretojev.jev.scoring import DecisionScores, _binary_entropy

PUBLIC_DEMO_ENDPOINT = "https://autotrust-jev-27b-demo.hf.space/api/decide"
PUBLIC_DEMO_MODEL = "autotrust/JEV-27B"


def validate_public_response(response: Any, revision: str) -> None:
    if not isinstance(response, dict):
        raise DecisionAPIError("Public demo response must be an object")
    if response.get("model") != PUBLIC_DEMO_MODEL or response.get("revision") != revision:
        raise DecisionAPIError("Public demo model/revision changed; refuse mixed or unknown weights")
    if response.get("kind") != "noul" or response.get("options") != ["false", "true"]:
        raise DecisionAPIError("Public demo returned an unexpected question type or option order")
    probabilities = response.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 2:
        raise DecisionAPIError("Public demo response lacks a binary probability distribution")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise DecisionAPIError("Public demo returned invalid probabilities")
    if abs(sum(probabilities) - 1) > 0.001:
        raise DecisionAPIError("Public demo probabilities do not sum to one")
    index = response.get("choice_index")
    if type(index) is not int or index not in (0, 1) or probabilities[index] != max(probabilities):
        raise DecisionAPIError("Public demo returned an inconsistent decision")
    if response.get("choice") != response["options"][index] or response.get("image_used") is not False:
        raise DecisionAPIError("Public demo returned an inconsistent option or unexpected image pipeline")
    elapsed = response.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise DecisionAPIError("Public demo returned invalid inference timing")


class PublicDemoJevClient:
    provider = "autotrust-public-demo"
    endpoint = PUBLIC_DEMO_ENDPOINT

    def __init__(self, *, revision: str, cache: DecisionCache,
                 max_http_calls: int = 400, minimum_interval: float = 1.0) -> None:
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("An explicit 40-character deployment revision is required")
        if max_http_calls <= 0 or not math.isfinite(minimum_interval) or minimum_interval < 1:
            raise ValueError("Use a positive request budget and an interval of at least one second")
        self.revision = revision
        self.model = f"{PUBLIC_DEMO_MODEL}@{revision}"
        self.cache = cache
        self.max_http_calls = max_http_calls
        self.minimum_interval = minimum_interval
        self.session = requests.Session()
        self._previous_start = 0.0
        self._calls = self._hits = self._requests = 0

    def stats(self) -> dict[str, int]:
        return {"requests": self._requests, "http_calls": self._calls, "cache_hits": self._hits}

    def decide(self, kind: str, state: Any, question: str, options=None) -> dict[str, Any]:
        if kind != "noul" or options is not None:
            raise ValueError("This bounded public pilot supports pointwise Noul only")
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("Public demo question must contain 1–1000 characters")
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        if len(state_text) > 6000:
            raise ValueError("Public demo state exceeds 6000 characters; do not truncate silently")
        payload = {"kind": kind, "state": state_text, "question": question}
        self._requests += 1
        cached = self.cache.get(self.provider, self.model, payload)
        if cached is not None:
            validate_public_response(cached, self.revision)
            self._hits += 1
            return cached
        if self._calls >= self.max_http_calls:
            raise DecisionAPIError("Public demo request budget exhausted")
        remaining = self.minimum_interval - (time.monotonic() - self._previous_start)
        if remaining > 0:
            time.sleep(remaining)
        self._previous_start = time.monotonic()
        self._calls += 1
        # No hidden retries: busy/error responses stop the pilot for inspection.
        response = self.session.post(self.endpoint, json=payload, timeout=(10, 60))
        response.raise_for_status()
        data = response.json()
        validate_public_response(data, self.revision)
        self.cache.put(self.provider, self.model, payload, data)
        if self._calls % 20 == 0:
            print(f"Validated real JEV calls: {self._calls}; cached: {self._hits}", flush=True)
        return data


def score_public_demo_noul(client: PublicDemoJevClient, state: Any,
                           candidates: list[dict[str, Any]],
                           objectives: Iterable[DecisionObjective]) -> DecisionScores:
    objectives = tuple(objectives)
    if isinstance(state, dict) and "candidates" in state:
        raise ValueError("Pointwise public scoring must hide competing candidates")
    responses = [client.decide(*request) for request in build_open_jev_noul_requests(state, candidates, objectives)]
    probabilities = np.asarray([row["probabilities"][1] for row in responses], dtype=np.float64).reshape(len(candidates), len(objectives))
    entropy = _binary_entropy(probabilities)
    result = DecisionScores(
        candidate_ids=tuple(str(row.get("candidate_id", row.get("item_id"))) for row in candidates),
        objective_names=tuple(objective.name for objective in objectives),
        probabilities=probabilities, confidence=1.0 - entropy, entropy=entropy,
        raw_response={"model": PUBLIC_DEMO_MODEL, "revision": client.revision,
                      "usage": {"decisions": len(responses)}, "responses": responses},
    )
    result.validate()
    return result
