from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from .cache import DecisionCache


class DecisionAPIError(RuntimeError):
    pass


class PermanentDecisionAPIError(DecisionAPIError):
    """Non-retryable authentication or request-validation failure."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0


class _HTTPDecisionClient:
    provider: str
    model: str

    def __init__(
        self,
        *,
        cache: DecisionCache | None = None,
        retry: RetryPolicy = RetryPolicy(),
        timeout_seconds: float = 120.0,
    ) -> None:
        self.cache = cache
        self.retry = retry
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        cached = self.cache.get(self.provider, self.model, payload) if self.cache else None
        if cached is not None:
            return cached

        delay = self.retry.initial_delay_seconds
        last_error: Exception | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        delay = min(float(retry_after), self.retry.max_delay_seconds)
                    raise DecisionAPIError(
                        f"temporary HTTP {response.status_code}: {response.text[:500]}"
                    )
                if response.status_code >= 400:
                    raise PermanentDecisionAPIError(
                        f"non-retryable HTTP {response.status_code}: {response.text[:1000]}"
                    )
                data = response.json()
                if self.cache:
                    self.cache.put(self.provider, self.model, payload, data)
                return data
            except PermanentDecisionAPIError:
                raise
            except (requests.RequestException, ValueError, DecisionAPIError) as exc:
                last_error = exc
                if attempt == self.retry.max_attempts:
                    break
                time.sleep(delay + random.random() * min(0.25, delay / 5))
                delay = min(delay * 2, self.retry.max_delay_seconds)
        raise DecisionAPIError(f"Decision API failed after retries: {last_error}") from last_error


class TypeSafeJevClient(_HTTPDecisionClient):
    """Client for the official TypeSafe `POST /v1/systemone` endpoint."""

    provider = "typesafe"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-1.13.0",
        cache: DecisionCache | None = None,
        retry: RetryPolicy = RetryPolicy(),
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(cache=cache, retry=retry, timeout_seconds=timeout_seconds)
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise ValueError("Set TYPESAFE_API_KEY (or pass api_key) before calling TypeSafe Jev")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def evaluate(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        payload = {"state": state, "model": self.model, "questions": questions}
        return self._post(
            f"{self.base_url}/v1/systemone",
            payload,
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )


class OpenJevClient(_HTTPDecisionClient):
    """Client for JEV-27B/JEV-27B-VL's open `/v1/decide` server."""

    provider = "open-jev"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "autotrust/JEV-27B",
        cache: DecisionCache | None = None,
        retry: RetryPolicy = RetryPolicy(),
        timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__(cache=cache, retry=retry, timeout_seconds=timeout_seconds)
        self.base_url = (base_url or os.environ.get("JEV_URL") or "http://localhost:8000").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.api_key = api_key or os.environ.get("JEV_API_KEY")
        self.model = model

    def decide(
        self,
        kind: str,
        state: Any,
        question: str,
        options: list[str] | None = None,
    ) -> dict[str, float]:
        if kind not in {"noul", "choice", "score"}:
            raise ValueError("kind must be noul, choice, or score")
        payload: dict[str, Any] = {"kind": kind, "state": state, "question": question}
        if kind == "choice":
            if not options or len(options) < 2:
                raise ValueError("choice requires at least two options")
            payload["options"] = options
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._post(f"{self.base_url}/v1/decide", payload, headers)
        if "options" not in data or "probabilities" not in data:
            raise DecisionAPIError(f"Malformed /v1/decide response: {data}")
        return {str(k): float(v) for k, v in zip(data["options"], data["probabilities"], strict=True)}

    def decide_many(self, requests_: Iterable[tuple[str, Any, str, list[str] | None]]) -> list[dict[str, float]]:
        # Deliberately sequential here. The caller can choose a concurrency level
        # suitable for its endpoint/rate limit; caching still prevents duplicate work.
        return [self.decide(kind, state, question, options) for kind, state, question, options in requests_]
