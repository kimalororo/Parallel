from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from aiohttp import web


@dataclass(frozen=True, slots=True)
class APIProfile:
    name: str
    min_delay_ms: float
    max_delay_ms: float
    error_rate: float = 0.0
    error_statuses: tuple[int, ...] = (500, 503)


class EmulatedAPIFarm:
    def __init__(
        self,
        profiles: list[APIProfile],
        *,
        host: str = "127.0.0.1",
        seed: int = 42,
    ) -> None:
        self.profiles = profiles
        self.host = host
        self.seed = seed
        self.urls: dict[str, str] = {}
        self._runners: list[web.AppRunner] = []

    async def start(self) -> dict[str, str]:
        for index, profile in enumerate(self.profiles):
            rng = random.Random(self.seed + index * 1009)
            app = web.Application()
            app.router.add_get("/data", self._make_handler(profile, rng))

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, 0)
            await site.start()

            sockets = site._server.sockets if site._server else []
            if not sockets:
                raise RuntimeError(f"cannot start emulator for {profile.name}")
            port = sockets[0].getsockname()[1]

            self._runners.append(runner)
            self.urls[profile.name] = f"http://{self.host}:{port}/data"

        return self.urls

    async def stop(self) -> None:
        for runner in reversed(self._runners):
            await runner.cleanup()
        self._runners.clear()
        self.urls.clear()

    def _make_handler(self, profile: APIProfile, rng: random.Random):
        async def handler(_: web.Request) -> web.Response:
            delay_ms = rng.uniform(profile.min_delay_ms, profile.max_delay_ms)
            await asyncio.sleep(delay_ms / 1000.0)

            if rng.random() < profile.error_rate:
                status = rng.choice(profile.error_statuses)
                return web.json_response(
                    {
                        "service": profile.name,
                        "ok": False,
                        "status": status,
                        "delay_ms": round(delay_ms, 2),
                        "ts": round(time.time(), 3),
                    },
                    status=status,
                )

            return web.json_response(
                {
                    "service": profile.name,
                    "ok": True,
                    "delay_ms": round(delay_ms, 2),
                    "ts": round(time.time(), 3),
                }
            )

        return handler

