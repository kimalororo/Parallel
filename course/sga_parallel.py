"""Sequential and multiprocessing-based Simple Genetic Algorithm experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


Bounds = tuple[float, float]


@dataclass(frozen=True)
class SGAConfig:
    objective: str = "rastrigin"
    dimensions: int = 20
    population_size: int = 100
    generations: int = 300
    bounds: Bounds = (-5.12, 5.12)
    processes: int = 1
    eval_repeats: int = 1
    crossover_rate: float = 0.85
    mutation_rate: float | None = None
    mutation_sigma: float | None = None
    tournament_size: int = 3
    elitism: int = 2
    seed: int = 2026


@dataclass
class SGAResult:
    objective: str
    dimensions: int
    population_size: int
    generations: int
    processes: int
    eval_repeats: int
    seed: int
    best_value: float
    runtime_sec: float
    avg_generation_ms: float
    best_vector: list[float]
    history: list[float]


_WORKER_OBJECTIVE = "rastrigin"
_WORKER_EVAL_REPEATS = 1


def sphere(x: Sequence[float]) -> float:
    return sum(value * value for value in x)


def rastrigin(x: Sequence[float]) -> float:
    n = len(x)
    return 10.0 * n + sum(value * value - 10.0 * math.cos(2.0 * math.pi * value) for value in x)


def rosenbrock(x: Sequence[float]) -> float:
    return sum(
        100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


OBJECTIVES: dict[str, Callable[[Sequence[float]], float]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}


def default_bounds(objective: str) -> Bounds:
    if objective in {"sphere", "rastrigin"}:
        return (-5.12, 5.12)
    if objective == "rosenbrock":
        return (-2.048, 2.048)
    raise ValueError(f"Unknown objective: {objective}")


def _init_worker(objective: str, eval_repeats: int) -> None:
    global _WORKER_OBJECTIVE, _WORKER_EVAL_REPEATS
    _WORKER_OBJECTIVE = objective
    _WORKER_EVAL_REPEATS = eval_repeats


def expensive_objective(x: Sequence[float], objective: str, eval_repeats: int) -> float:
    """Repeat the same objective to model expensive black-box fitness evaluation."""
    func = OBJECTIVES[objective]
    total = 0.0
    for _ in range(eval_repeats):
        total += func(x)
    return total / eval_repeats


def _worker_eval(candidate: Sequence[float]) -> float:
    return expensive_objective(candidate, _WORKER_OBJECTIVE, _WORKER_EVAL_REPEATS)


def initialize_population(
    population_size: int,
    dimensions: int,
    bounds: Bounds,
    rng: random.Random,
) -> list[list[float]]:
    lower, upper = bounds
    return [[rng.uniform(lower, upper) for _ in range(dimensions)] for _ in range(population_size)]


def evaluate_population(
    population: Sequence[Sequence[float]],
    config: SGAConfig,
    pool: mp.pool.Pool | None = None,
) -> list[float]:
    if pool is None:
        return [expensive_objective(candidate, config.objective, config.eval_repeats) for candidate in population]
    chunksize = max(1, len(population) // (config.processes * 4))
    return list(pool.map(_worker_eval, population, chunksize=chunksize))


def tournament_select(values: Sequence[float], rng: random.Random, tournament_size: int) -> int:
    best_index = rng.randrange(len(values))
    best_value = values[best_index]
    for _ in range(tournament_size - 1):
        index = rng.randrange(len(values))
        value = values[index]
        if value < best_value:
            best_index = index
            best_value = value
    return best_index


def crossover(
    parent_a: Sequence[float],
    parent_b: Sequence[float],
    bounds: Bounds,
    rng: random.Random,
    crossover_rate: float,
) -> tuple[list[float], list[float]]:
    if rng.random() > crossover_rate:
        return list(parent_a), list(parent_b)

    lower, upper = bounds
    child_a: list[float] = []
    child_b: list[float] = []
    for gene_a, gene_b in zip(parent_a, parent_b):
        alpha = rng.random()
        first = alpha * gene_a + (1.0 - alpha) * gene_b
        second = alpha * gene_b + (1.0 - alpha) * gene_a
        child_a.append(min(upper, max(lower, first)))
        child_b.append(min(upper, max(lower, second)))
    return child_a, child_b


def mutate(
    individual: list[float],
    bounds: Bounds,
    rng: random.Random,
    mutation_rate: float,
    mutation_sigma: float,
) -> None:
    lower, upper = bounds
    for index, value in enumerate(individual):
        if rng.random() < mutation_rate:
            mutated = value + rng.gauss(0.0, mutation_sigma)
            individual[index] = min(upper, max(lower, mutated))


def make_next_generation(
    population: Sequence[Sequence[float]],
    values: Sequence[float],
    config: SGAConfig,
    rng: random.Random,
) -> list[list[float]]:
    ranked = sorted(range(len(population)), key=lambda index: values[index])
    next_population = [list(population[index]) for index in ranked[: config.elitism]]

    mutation_rate = config.mutation_rate if config.mutation_rate is not None else 1.0 / config.dimensions
    lower, upper = config.bounds
    mutation_sigma = config.mutation_sigma if config.mutation_sigma is not None else 0.08 * (upper - lower)

    while len(next_population) < config.population_size:
        parent_a = population[tournament_select(values, rng, config.tournament_size)]
        parent_b = population[tournament_select(values, rng, config.tournament_size)]
        child_a, child_b = crossover(parent_a, parent_b, config.bounds, rng, config.crossover_rate)
        mutate(child_a, config.bounds, rng, mutation_rate, mutation_sigma)
        mutate(child_b, config.bounds, rng, mutation_rate, mutation_sigma)
        next_population.append(child_a)
        if len(next_population) < config.population_size:
            next_population.append(child_b)

    return next_population


def run_sga(config: SGAConfig) -> SGAResult:
    if config.objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {config.objective}")
    if config.processes < 1:
        raise ValueError("processes must be >= 1")
    if config.population_size <= config.elitism:
        raise ValueError("population_size must be greater than elitism")

    rng = random.Random(config.seed)
    population = initialize_population(config.population_size, config.dimensions, config.bounds, rng)
    history: list[float] = []
    generation_times: list[float] = []
    best_vector: list[float] = []

    start = time.perf_counter()

    if config.processes == 1:
        values = evaluate_population(population, config)
        for _ in range(config.generations):
            gen_start = time.perf_counter()
            best_index = min(range(len(values)), key=lambda index: values[index])
            best_vector = list(population[best_index])
            history.append(values[best_index])
            population = make_next_generation(population, values, config, rng)
            values = evaluate_population(population, config)
            generation_times.append(time.perf_counter() - gen_start)
        best_index = min(range(len(values)), key=lambda index: values[index])
        best_vector = list(population[best_index])
        history.append(values[best_index])
    else:
        with mp.Pool(
            processes=config.processes,
            initializer=_init_worker,
            initargs=(config.objective, config.eval_repeats),
        ) as pool:
            values = evaluate_population(population, config, pool=pool)
            for _ in range(config.generations):
                gen_start = time.perf_counter()
                best_index = min(range(len(values)), key=lambda index: values[index])
                best_vector = list(population[best_index])
                history.append(values[best_index])
                population = make_next_generation(population, values, config, rng)
                values = evaluate_population(population, config, pool=pool)
                generation_times.append(time.perf_counter() - gen_start)
            best_index = min(range(len(values)), key=lambda index: values[index])
            best_vector = list(population[best_index])
            history.append(values[best_index])

    runtime = time.perf_counter() - start
    return SGAResult(
        objective=config.objective,
        dimensions=config.dimensions,
        population_size=config.population_size,
        generations=config.generations,
        processes=config.processes,
        eval_repeats=config.eval_repeats,
        seed=config.seed,
        best_value=min(history),
        runtime_sec=runtime,
        avg_generation_ms=(sum(generation_times) / len(generation_times)) * 1000.0,
        best_vector=best_vector,
        history=history,
    )


def write_history_csv(results: Iterable[SGAResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "generation", "best_value"])
        for run_id, result in enumerate(results, start=1):
            for generation, best_value in enumerate(result.history):
                writer.writerow([run_id, generation, f"{best_value:.12g}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Simple Genetic Algorithm optimization.")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="rastrigin")
    parser.add_argument("--dimensions", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SGAConfig(
        objective=args.objective,
        dimensions=args.dimensions,
        population_size=args.population_size,
        generations=args.generations,
        bounds=default_bounds(args.objective),
        processes=args.processes,
        eval_repeats=args.eval_repeats,
        seed=args.seed,
    )
    result = run_sga(config)
    payload = asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()
