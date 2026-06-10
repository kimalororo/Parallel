from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from urllib.parse import urlparse


def endpoint_key(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    return f"{parsed.netloc}{path}"


@dataclass(slots=True)
class RequestSample:
    success: bool
    elapsed_ms: float
    timeout: bool = False
    status: int | None = None


class EndpointStats:
    def __init__(self, window_size: int, initial_concurrency: int, hard_limit: int) -> None:
        self.window_size = window_size
        self.hard_limit = max(1, hard_limit)
        self.current_concurrency = min(max(1, initial_concurrency), self.hard_limit)
        self.samples: deque[RequestSample] = deque(maxlen=window_size)
        self.total_requests = 0
        self.total_successful = 0
        self.total_failed = 0

    def update_limits(self, initial_concurrency: int, hard_limit: int) -> None:
        self.hard_limit = max(1, hard_limit)
        if self.total_requests == 0:
            self.current_concurrency = min(max(1, initial_concurrency), self.hard_limit)
        else:
            self.current_concurrency = min(self.current_concurrency, self.hard_limit)

    def record(self, sample: RequestSample) -> None:
        self.samples.append(sample)
        self.total_requests += 1
        if sample.success:
            self.total_successful += 1
        else:
            self.total_failed += 1

    @property
    def success_rate(self) -> float:
        if not self.samples:
            return 0.0
        successes = sum(1 for sample in self.samples if sample.success)
        return successes / len(self.samples)

    @property
    def avg_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.elapsed_ms for sample in self.samples) / len(self.samples)

    @property
    def timeout_count(self) -> int:
        return sum(1 for sample in self.samples if sample.timeout)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "success_rate": round(self.success_rate, 3),
            "avg_ms": round(self.avg_ms, 2),
            "adjusted_concurrency": self.current_concurrency,
            "window_size": len(self.samples),
            "total_requests": self.total_requests,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
        }


class AdaptiveLoadBalancer:
    def __init__(
        self,
        window_size: int = 20,
        min_samples: int = 4,
        error_threshold: float = 0.30,
        slow_threshold_ms: float = 2000.0,
        fast_threshold_ms: float = 500.0,
        adaptive_multiplier: int = 2,
        absolute_limit: int = 12,
    ) -> None:
        self.window_size = window_size
        self.min_samples = min_samples
        self.error_threshold = error_threshold
        self.slow_threshold_ms = slow_threshold_ms
        self.fast_threshold_ms = fast_threshold_ms
        self.adaptive_multiplier = adaptive_multiplier
        self.absolute_limit = absolute_limit
        self._stats: dict[str, EndpointStats] = {}
        self._lock = RLock()

    def hard_limit_for(self, initial_concurrency: int) -> int:
        initial = max(1, initial_concurrency)
        return min(self.absolute_limit, max(initial, initial * self.adaptive_multiplier))

    def suggested_concurrency(self, url: str, initial_concurrency: int, hard_limit: int) -> int:
        key = endpoint_key(url)
        with self._lock:
            stats = self._ensure(key, initial_concurrency, hard_limit)
            return stats.current_concurrency

    def record(
        self,
        url: str,
        success: bool,
        elapsed_ms: float,
        *,
        timeout: bool = False,
        status: int | None = None,
        initial_concurrency: int = 3,
        hard_limit: int | None = None,
    ) -> None:
        key = endpoint_key(url)
        limit = hard_limit if hard_limit is not None else self.hard_limit_for(initial_concurrency)
        with self._lock:
            stats = self._ensure(key, initial_concurrency, limit)
            stats.record(RequestSample(success, elapsed_ms, timeout=timeout, status=status))
            self._adjust(stats)

    def snapshot_for_urls(self, urls: list[str]) -> dict[str, dict[str, float | int]]:
        with self._lock:
            keys = [endpoint_key(url) for url in urls]
            return {
                key: self._stats[key].snapshot()
                for key in dict.fromkeys(keys)
                if key in self._stats
            }

    def snapshot_all(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {key: stats.snapshot() for key, stats in self._stats.items()}

    def _ensure(self, key: str, initial_concurrency: int, hard_limit: int) -> EndpointStats:
        if key not in self._stats:
            self._stats[key] = EndpointStats(self.window_size, initial_concurrency, hard_limit)
        else:
            self._stats[key].update_limits(initial_concurrency, hard_limit)
        return self._stats[key]

    def _adjust(self, stats: EndpointStats) -> None:
        if len(stats.samples) < self.min_samples:
            return

        failure_rate = 1.0 - stats.success_rate
        if failure_rate > self.error_threshold or stats.avg_ms > self.slow_threshold_ms:
            stats.current_concurrency = max(1, stats.current_concurrency - 1)
            return

        if stats.success_rate >= 0.90 and stats.avg_ms < self.fast_threshold_ms:
            stats.current_concurrency = min(stats.hard_limit, stats.current_concurrency + 1)

