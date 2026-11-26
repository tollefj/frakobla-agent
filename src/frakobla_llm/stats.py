from __future__ import annotations

import time
from dataclasses import dataclass, replace

from frakobla_llm.client import Completion


@dataclass
class Usage:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    reasoning_tokens: int = 0
    inference_seconds: float = 0.0


class Stats:
    def __init__(self) -> None:
        self.total = Usage()
        self._live = Usage()
        self._started: float | None = None

    def begin(self) -> None:
        self._live = Usage()
        self._started = time.monotonic()

    def token(self, kind: str) -> None:
        self._live.generated_tokens += 1
        if kind == "reasoning":
            self._live.reasoning_tokens += 1

    def end(self, completion: Completion | None) -> None:
        if self._started is None:
            return
        live = self._live
        live.inference_seconds = time.monotonic() - self._started
        if completion is not None:
            _apply_server_counts(live, completion)
        self.total = _sum(self.total, live)
        self._live = Usage()
        self._started = None

    def current(self) -> Usage:
        if self._started is None:
            return replace(self.total)
        live = replace(self._live, inference_seconds=time.monotonic() - self._started)
        return _sum(self.total, live)


def energy_kwh(usage: Usage, watts: float) -> float:
    return watts * usage.inference_seconds / 3_600_000


def _apply_server_counts(usage: Usage, completion: Completion) -> None:
    timings, counts = completion.timings, completion.usage
    if timings.get("prompt_ms") is not None and timings.get("predicted_ms") is not None:
        usage.inference_seconds = (timings["prompt_ms"] + timings["predicted_ms"]) / 1000
    usage.prompt_tokens = timings.get("prompt_n") or counts.get("prompt_tokens") or 0
    usage.generated_tokens = timings.get("predicted_n") or counts.get("completion_tokens") or usage.generated_tokens
    details = counts.get("completion_tokens_details") or {}
    usage.reasoning_tokens = details.get("reasoning_tokens") or usage.reasoning_tokens


def _sum(a: Usage, b: Usage) -> Usage:
    return Usage(
        a.prompt_tokens + b.prompt_tokens,
        a.generated_tokens + b.generated_tokens,
        a.reasoning_tokens + b.reasoning_tokens,
        a.inference_seconds + b.inference_seconds,
    )
