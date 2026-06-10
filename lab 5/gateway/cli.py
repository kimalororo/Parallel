from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import web

from .api import create_app
from .benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 5 asynchronous aggregation gateway")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="start HTTP aggregation gateway")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    test_parser = subparsers.add_parser("test", help="run built-in benchmark scenarios")
    test_parser.add_argument("--scenario", default="all", choices=["all", "stable", "slow", "unstable"])
    test_parser.add_argument("--repeats", type=int, default=10)
    test_parser.add_argument("--output", default="report/")

    args = parser.parse_args()

    if args.command == "serve":
        web.run_app(create_app(), host=args.host, port=args.port)
        return 0

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    result = asyncio.run(run_benchmark(args.scenario, args.repeats, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

