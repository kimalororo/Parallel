from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


ALLOWED_STRATEGIES = {"fixed", "timeout_race", "adaptive"}


@dataclass(slots=True)
class AggregationRequest:
    urls: list[str]
    strategy: str = "fixed"
    max_concurrent: int = 3
    timeout_sec: float = 5.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AggregationRequest":
        urls = payload.get("urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError("urls must be a non-empty list")

        normalized_urls: list[str] = []
        for value in urls:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("each url must be a non-empty string")
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"invalid HTTP URL: {value}")
            normalized_urls.append(value.strip())

        strategy = payload.get("strategy", "fixed")
        if strategy not in ALLOWED_STRATEGIES:
            allowed = ", ".join(sorted(ALLOWED_STRATEGIES))
            raise ValueError(f"strategy must be one of: {allowed}")

        max_concurrent = int(payload.get("max_concurrent", 3))
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        timeout_sec = float(payload.get("timeout_sec", 5))
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")

        return cls(
            urls=normalized_urls,
            strategy=strategy,
            max_concurrent=max_concurrent,
            timeout_sec=timeout_sec,
        )

