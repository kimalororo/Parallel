from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .emulators import APIProfile, EmulatedAPIFarm
from .models import AggregationRequest
from .strategies import Aggregator


STRATEGIES = ("fixed", "timeout_race", "adaptive")


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    title: str
    description: str
    profiles: tuple[APIProfile, ...]
    timeout_sec: float = 1.2
    max_concurrent: int = 3
    requests_per_api: int = 3


SCENARIOS: dict[str, Scenario] = {
    "stable": Scenario(
        name="stable",
        title="Scenario A: stable",
        description="All APIs respond within 50-180 ms and do not return errors.",
        profiles=(
            APIProfile("stable-a", 50, 160),
            APIProfile("stable-b", 70, 180),
            APIProfile("stable-c", 60, 170),
        ),
    ),
    "slow": Scenario(
        name="slow",
        title="Scenario B: slow",
        description="One API responds after 3-5 seconds, while the others are fast.",
        profiles=(
            APIProfile("fast-a", 70, 180),
            APIProfile("fast-b", 80, 220),
            APIProfile("slow-c", 3000, 5000),
        ),
    ),
    "unstable": Scenario(
        name="unstable",
        title="Scenario C: unstable",
        description="Some APIs return HTTP 500/503 with probability 30-50%.",
        profiles=(
            APIProfile("stable-a", 80, 220),
            APIProfile("flaky-b", 120, 360, error_rate=0.35),
            APIProfile("flaky-c", 150, 480, error_rate=0.50),
        ),
    ),
}


async def run_benchmark(scenario_name: str, repeats: int, output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected = _select_scenarios(scenario_name)
    rows: list[dict[str, Any]] = []
    adaptive_history: list[dict[str, Any]] = []
    scenario_meta: list[dict[str, Any]] = []

    for scenario in selected:
        farm = EmulatedAPIFarm(list(scenario.profiles), seed=100 + len(rows))
        try:
            urls = await farm.start()
            url_batch = _make_url_batch(scenario, urls)
            scenario_meta.append(
                {
                    "name": scenario.name,
                    "title": scenario.title,
                    "description": scenario.description,
                    "timeout_sec": scenario.timeout_sec,
                    "max_concurrent": scenario.max_concurrent,
                    "requests_per_api": scenario.requests_per_api,
                    "emulators": {
                        profile.name: {
                            "url": urls[profile.name],
                            "min_delay_ms": profile.min_delay_ms,
                            "max_delay_ms": profile.max_delay_ms,
                            "error_rate": profile.error_rate,
                        }
                        for profile in scenario.profiles
                    },
                }
            )

            for strategy in STRATEGIES:
                aggregator = Aggregator()
                for repeat in range(1, repeats + 1):
                    request = AggregationRequest(
                        urls=url_batch,
                        strategy=strategy,
                        max_concurrent=scenario.max_concurrent,
                        timeout_sec=scenario.timeout_sec,
                    )
                    response = await aggregator.aggregate(request)
                    summary = response["summary"]
                    timeouts = sum(1 for item in response["results"] if item.get("timeout"))
                    row = {
                        "scenario": scenario.name,
                        "strategy": strategy,
                        "repeat": repeat,
                        "total_time_ms": summary["total_time_ms"],
                        "successful": summary["successful"],
                        "failed": summary["failed"],
                        "timeouts": timeouts,
                        "concurrent_used": summary["concurrent_used"],
                    }
                    rows.append(row)

                    if strategy == "adaptive":
                        for api_key, stats in response.get("adaptive_stats", {}).items():
                            adaptive_history.append(
                                {
                                    "scenario": scenario.name,
                                    "repeat": repeat,
                                    "api": api_key,
                                    "adjusted_concurrency": stats["adjusted_concurrency"],
                                    "success_rate": stats["success_rate"],
                                    "avg_ms": stats["avg_ms"],
                                }
                            )
        finally:
            await farm.stop()

    _write_json(output_path / "results.json", rows, adaptive_history, scenario_meta)
    _write_csv(output_path / "results.csv", rows)
    _write_csv(output_path / "adaptive_history.csv", adaptive_history)
    _plot_average_time(output_path / "avg_time_by_strategy.png", rows)
    _plot_success_failure(output_path / "success_failure_by_strategy.png", rows)
    _plot_adaptive_concurrency(output_path / "adaptive_concurrency.png", adaptive_history)
    _write_report(output_path / "report.md", rows, adaptive_history, scenario_meta, repeats)

    return {
        "output_dir": str(output_path.resolve()),
        "rows": len(rows),
        "adaptive_points": len(adaptive_history),
        "scenarios": [scenario.name for scenario in selected],
    }


def _select_scenarios(name: str) -> list[Scenario]:
    if name == "all":
        return list(SCENARIOS.values())
    if name not in SCENARIOS:
        available = ", ".join(["all", *SCENARIOS])
        raise ValueError(f"unknown scenario '{name}', expected one of: {available}")
    return [SCENARIOS[name]]


def _make_url_batch(scenario: Scenario, urls: dict[str, str]) -> list[str]:
    result: list[str] = []
    for _ in range(scenario.requests_per_api):
        for profile in scenario.profiles:
            result.append(urls[profile.name])
    return result


def _write_json(
    path: Path,
    rows: list[dict[str, Any]],
    adaptive_history: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scenarios": scenarios,
        "results": rows,
        "adaptive_history": adaptive_history,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["strategy"])].append(row)
    return grouped


def _plot_average_time(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped = _group_rows(rows)
    scenarios = _ordered_unique(row["scenario"] for row in rows)
    width = 0.24
    x_positions = list(range(len(scenarios)))
    colors = {"fixed": "#2f6f9f", "timeout_race": "#d95f02", "adaptive": "#2a9d55"}

    plt.figure(figsize=(10, 5))
    for index, strategy in enumerate(STRATEGIES):
        values = [
            mean(row["total_time_ms"] for row in grouped.get((scenario, strategy), []))
            if grouped.get((scenario, strategy))
            else 0
            for scenario in scenarios
        ]
        offsets = [x + (index - 1) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=strategy, color=colors[strategy])

    plt.title("Average aggregation time by strategy")
    plt.ylabel("Milliseconds")
    plt.xticks(x_positions, scenarios)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_success_failure(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped = _group_rows(rows)
    labels = [f"{scenario}\n{strategy}" for scenario, strategy in grouped]
    successes = [sum(row["successful"] for row in items) for items in grouped.values()]
    failures = [sum(row["failed"] for row in items) for items in grouped.values()]
    x_positions = list(range(len(labels)))

    plt.figure(figsize=(12, 5.5))
    plt.bar(x_positions, successes, label="successful", color="#2a9d55")
    plt.bar(x_positions, failures, bottom=successes, label="failed", color="#c44536")
    plt.title("Successful and failed external requests")
    plt.ylabel("Requests")
    plt.xticks(x_positions, labels)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_adaptive_concurrency(path: Path, adaptive_history: list[dict[str, Any]]) -> None:
    unstable_rows = [row for row in adaptive_history if row["scenario"] == "unstable"]
    rows = unstable_rows or adaptive_history

    plt.figure(figsize=(10, 5))
    if rows:
        by_api: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_api[row["api"]].append(row)
        for api, items in by_api.items():
            ordered = sorted(items, key=lambda item: item["repeat"])
            plt.plot(
                [item["repeat"] for item in ordered],
                [item["adjusted_concurrency"] for item in ordered],
                marker="o",
                label=api,
            )
    else:
        plt.text(0.5, 0.5, "No adaptive data", ha="center", va="center")

    plt.title("Adaptive concurrency over time")
    plt.xlabel("Repeat")
    plt.ylabel("Adjusted concurrency")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    adaptive_history: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    repeats: int,
) -> None:
    grouped = _group_rows(rows)
    table_lines = [
        "| Scenario | Strategy | Avg time, ms | Avg successful | Avg failed | Avg timeouts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in _ordered_unique(row["scenario"] for row in rows):
        for strategy in STRATEGIES:
            items = grouped.get((scenario, strategy), [])
            if not items:
                continue
            table_lines.append(
                "| {scenario} | {strategy} | {avg_time:.2f} | {avg_success:.2f} | "
                "{avg_failed:.2f} | {avg_timeouts:.2f} |".format(
                    scenario=scenario,
                    strategy=strategy,
                    avg_time=mean(row["total_time_ms"] for row in items),
                    avg_success=mean(row["successful"] for row in items),
                    avg_failed=mean(row["failed"] for row in items),
                    avg_timeouts=mean(row["timeouts"] for row in items),
                )
            )

    adaptive_summary = _adaptive_summary(adaptive_history)
    scenario_text = "\n".join(
        f"- `{item['name']}`: {item['description']} Timeout: {item['timeout_sec']} s, "
        f"requests per API: {item['requests_per_api']}."
        for item in scenarios
    )

    report = f"""# Отчёт по лабораторной работе 5

## Тема

Асинхронный шлюз агрегации данных с динамической балансировкой.

## Выбранные внешние API

В работе используются локальные эмуляторы HTTP API на `aiohttp.web`. Такой вариант выбран вместо публичных API, потому что он воспроизводим: задержки, таймауты и вероятность ошибок контролируются кодом тестовой подсистемы.

{scenario_text}

## Архитектура

- `gateway/api.py` — HTTP-шлюз с endpoint `POST /aggregate`.
- `gateway/strategies.py` — стратегии `fixed`, `timeout_race` и `adaptive`.
- `gateway/stats.py` — скользящее окно статистики и адаптивная корректировка concurrency.
- `gateway/emulators.py` — локальные API-эмуляторы.
- `gateway/benchmark.py` — CLI-бенчмарк, сохранение JSON/CSV и построение графиков.
- `tests/test_adaptive.py` — модульные тесты логики адаптации.

## Адаптивная стратегия

Для каждого URL ведётся скользящее окно последних запросов. В окне считаются доля успешных ответов и среднее время ответа. Если доля ошибок выше 30% или среднее время ответа превышает 2000 мс, concurrency для этого API уменьшается на 1, но не ниже 1. Если success rate не ниже 90%, а среднее время ниже 500 мс, concurrency осторожно увеличивается на 1 до внутреннего верхнего лимита.

## Результаты

Тестовый прогон выполнен с `repeats={repeats}`. Для каждого вызова агрегировалось по 3 запроса к каждому из трёх API.

{chr(10).join(table_lines)}

Графики сохранены рядом с этим отчётом:

- `avg_time_by_strategy.png` — среднее время выполнения по стратегиям.
- `adaptive_concurrency.png` — изменение concurrency для адаптивной стратегии.
- `success_failure_by_strategy.png` — успешные и неуспешные запросы.

## Поведение adaptive

{adaptive_summary}

## Анализ

На стабильных API агрессивные стратегии обычно выигрывают по времени, потому что все ответы быстрые и нет смысла искусственно ограничивать запросы. На медленном сценарии `fixed` страдает от очередей: медленные запросы занимают слоты, а следующие ждут освобождения semaphore. `timeout_race` быстрее отсекает медленные API, так как запускает все запросы сразу. `adaptive` занимает промежуточную позицию: после накопления статистики он снижает нагрузку на проблемные URL и защищает общий пул запросов.

В нестабильном сценарии адаптивная стратегия уменьшает concurrency у API с частыми 500/503, поэтому сервис тратит меньше одновременных слотов на заведомо проблемные источники. Ограничение подхода — адаптации нужно накопить несколько измерений, поэтому первые запросы выполняются с начальной настройкой.

## Инструкция по запуску

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python main.py serve --host 127.0.0.1 --port 8000
```

Проверка endpoint:

```bash
curl -X POST http://127.0.0.1:8000/aggregate -H "Content-Type: application/json" -d "{{\\"urls\\":[\\"https://jsonplaceholder.typicode.com/todos/1\\"],\\"strategy\\":\\"adaptive\\"}}"
```

Запуск встроенной тестовой системы:

```bash
python main.py test --scenario all --repeats 10 --output report/
```

Модульные тесты:

```bash
pytest
```

## Выводы

Асинхронность позволяет выполнять независимые HTTP-запросы без последовательного ожидания каждого API. Фиксированная стратегия предсказуема, но плохо реагирует на медленные источники. `timeout_race` хорош для минимизации времени ожидания, но может создавать всплеск нагрузки. `adaptive` добавляет обратную связь: он использует реальные измерения времени ответа и ошибок, чтобы постепенно перераспределять доступную параллельность.
"""
    path.write_text(report, encoding="utf-8")


def _adaptive_summary(adaptive_history: list[dict[str, Any]]) -> str:
    if not adaptive_history:
        return "Данные adaptive отсутствуют."

    by_api: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adaptive_history:
        by_api[f"{row['scenario']}:{row['api']}"].append(row)

    lines: list[str] = []
    for key, items in sorted(by_api.items()):
        ordered = sorted(items, key=lambda item: item["repeat"])
        first = ordered[0]["adjusted_concurrency"]
        last = ordered[-1]["adjusted_concurrency"]
        lines.append(f"- `{key}`: concurrency {first} -> {last}.")
    return "\n".join(lines)


def _ordered_unique(values) -> list[Any]:
    return list(dict.fromkeys(values))

