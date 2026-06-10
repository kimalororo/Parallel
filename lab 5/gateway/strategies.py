from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import aiohttp

from .models import AggregationRequest
from .stats import AdaptiveLoadBalancer, endpoint_key


class Aggregator:
    def __init__(self, balancer: AdaptiveLoadBalancer | None = None) -> None:
        self.balancer = balancer or AdaptiveLoadBalancer()

    async def aggregate(self, request: AggregationRequest) -> dict[str, Any]:
        request_started = time.perf_counter()
        hard_limit = self.balancer.hard_limit_for(request.max_concurrent)

        async with aiohttp.ClientSession() as session:
            if request.strategy == "fixed":
                results, concurrent_used = await self._fixed(session, request, hard_limit)
            elif request.strategy == "timeout_race":
                results, concurrent_used = await self._timeout_race(session, request, hard_limit)
            else:
                results, concurrent_used = await self._adaptive(session, request, hard_limit)

        total_time_ms = round((time.perf_counter() - request_started) * 1000, 2)
        successful = sum(1 for item in results if self._is_success(item))
        failed = len(results) - successful

        response: dict[str, Any] = {
            "request_id": str(uuid4()),
            "results": results,
            "summary": {
                "total": len(results),
                "successful": successful,
                "failed": failed,
                "total_time_ms": total_time_ms,
                "strategy_used": request.strategy,
                "concurrent_used": concurrent_used,
            },
        }

        if request.strategy == "adaptive":
            response["adaptive_stats"] = self.balancer.snapshot_for_urls(request.urls)

        return response

    async def _fixed(
        self,
        session: aiohttp.ClientSession,
        request: AggregationRequest,
        hard_limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def worker(url: str) -> dict[str, Any]:
            async with semaphore:
                return await self._fetch_and_record(session, url, request, hard_limit)

        return await asyncio.gather(*(worker(url) for url in request.urls)), request.max_concurrent

    async def _timeout_race(
        self,
        session: aiohttp.ClientSession,
        request: AggregationRequest,
        hard_limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        tasks = [
            asyncio.create_task(self._fetch_and_record(session, url, request, hard_limit))
            for url in request.urls
        ]
        return await asyncio.gather(*tasks), len(tasks)

    async def _adaptive(
        self,
        session: aiohttp.ClientSession,
        request: AggregationRequest,
        hard_limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        concurrency_plan: dict[str, int] = {}
        for url in request.urls:
            key = endpoint_key(url)
            concurrency_plan[key] = self.balancer.suggested_concurrency(
                url,
                request.max_concurrent,
                hard_limit,
            )

        per_api_semaphores = {
            key: asyncio.Semaphore(limit) for key, limit in concurrency_plan.items()
        }
        global_semaphore = asyncio.Semaphore(hard_limit)

        async def worker(url: str) -> dict[str, Any]:
            key = endpoint_key(url)
            async with global_semaphore:
                async with per_api_semaphores[key]:
                    return await self._fetch_and_record(session, url, request, hard_limit)

        concurrent_used = min(hard_limit, sum(concurrency_plan.values()))
        return await asyncio.gather(*(worker(url) for url in request.urls)), concurrent_used

    async def _fetch_and_record(
        self,
        session: aiohttp.ClientSession,
        url: str,
        request: AggregationRequest,
        hard_limit: int,
    ) -> dict[str, Any]:
        result, success = await self._fetch_url(session, url, request.timeout_sec)
        self.balancer.record(
            url,
            success,
            float(result["elapsed_ms"]),
            timeout=bool(result.get("timeout", False)),
            status=result.get("status"),
            initial_concurrency=request.max_concurrent,
            hard_limit=hard_limit,
        )
        return result

    async def _fetch_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
        timeout_sec: float,
    ) -> tuple[dict[str, Any], bool]:
        started = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=timeout_sec)

        try:
            async with session.get(url, timeout=timeout) as response:
                data = await self._read_payload(response)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                success = 200 <= response.status < 400
                item: dict[str, Any] = {
                    "url": url,
                    "status": response.status,
                    "elapsed_ms": elapsed_ms,
                }
                if success:
                    item["data"] = data
                else:
                    item["error"] = response.reason or "HTTP error"
                    item["data"] = data
                return item, success
        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return {"url": url, "timeout": True, "elapsed_ms": elapsed_ms}, False
        except aiohttp.ClientError as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return {"url": url, "error": str(exc), "elapsed_ms": elapsed_ms}, False

    @staticmethod
    async def _read_payload(response: aiohttp.ClientResponse) -> Any:
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            return await response.json(content_type=None)
        return await response.text()

    @staticmethod
    def _is_success(item: dict[str, Any]) -> bool:
        status = item.get("status")
        return isinstance(status, int) and 200 <= status < 400

