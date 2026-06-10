from __future__ import annotations

from aiohttp import web

from .models import AggregationRequest
from .strategies import Aggregator


def create_app(aggregator: Aggregator | None = None) -> web.Application:
    app = web.Application()
    app["aggregator"] = aggregator or Aggregator()

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def aggregate(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        try:
            aggregation_request = AggregationRequest.from_mapping(payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=422)

        result = await request.app["aggregator"].aggregate(aggregation_request)
        return web.json_response(result)

    app.router.add_get("/health", health)
    app.router.add_post("/aggregate", aggregate)
    return app

