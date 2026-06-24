"""Run reproducible experiments for the SGA coursework report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sga_parallel import SGAConfig, default_bounds, run_sga


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOT_CONVERGENCE_CONFIGS = {
    (50, 100),
    (50, 500),
    (100, 500),
    (500, 500),
    (500, 1000),
}


def make_config(
    objective: str,
    population_size: int,
    generations: int,
    processes: int,
    seed: int,
    eval_repeats: int,
    dimensions: int = 20,
) -> SGAConfig:
    return SGAConfig(
        objective=objective,
        dimensions=dimensions,
        population_size=population_size,
        generations=generations,
        bounds=default_bounds(objective),
        processes=processes,
        eval_repeats=eval_repeats,
        seed=seed,
    )


def run_and_label(label: str, config: SGAConfig) -> dict[str, object]:
    print(
        f"{label}: objective={config.objective}, pop={config.population_size}, "
        f"gen={config.generations}, p={config.processes}, seed={config.seed}"
    )
    started = time.perf_counter()
    result = run_sga(config)
    elapsed = time.perf_counter() - started
    row = asdict(result)
    row.pop("best_vector")
    row.pop("history")
    row["label"] = label
    row["wall_clock_sec"] = elapsed
    row["final_best_value"] = result.history[-1]
    return row | {"history": result.history}


def summarize_by_process(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["processes"]), []).append(row)

    summaries: list[dict[str, object]] = []
    baseline = statistics.mean(float(row["runtime_sec"]) for row in grouped[1])
    for processes in sorted(grouped):
        runtimes = [float(row["runtime_sec"]) for row in grouped[processes]]
        best_values = [float(row["best_value"]) for row in grouped[processes]]
        mean_runtime = statistics.mean(runtimes)
        speedup = baseline / mean_runtime
        summaries.append(
            {
                "processes": processes,
                "runs": len(runtimes),
                "mean_runtime_sec": mean_runtime,
                "stdev_runtime_sec": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
                "mean_best_value": statistics.mean(best_values),
                "speedup": speedup,
                "efficiency_pct": speedup / processes * 100.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def font(size: int = 16) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    y_label: str,
    x_label: str,
) -> None:
    left, top, right, bottom = box
    axis_color = (45, 55, 72)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    draw.text((left, top - 36), title, font=font(22), fill=(20, 33, 61))
    draw.text((left, bottom + 18), x_label, font=font(13), fill=(70, 70, 70))
    draw.text((20, top + 10), y_label, font=font(13), fill=(70, 70, 70))


def scale_points(
    xs: list[float],
    ys: list[float],
    box: tuple[int, int, int, int],
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    x_min = min(xs) if x_min is None else x_min
    x_max = max(xs) if x_max is None else x_max
    y_min = min(ys) if y_min is None else y_min
    y_max = max(ys) if y_max is None else y_max
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, ys):
        px = left + int((x - x_min) / (x_max - x_min) * (right - left))
        py = bottom - int((y - y_min) / (y_max - y_min) * (bottom - top))
        points.append((px, py))
    return points


def plot_convergence(selected: list[dict[str, object]], output_path: Path) -> None:
    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    box = (95, 105, 1110, 610)
    draw_axes(draw, box, "Convergence on Rastrigin, log10(best + 1e-9)", "log objective", "generation")

    colors = [(46, 116, 181), (31, 78, 121), (122, 90, 0), (155, 28, 28), (53, 94, 59)]
    prepared: list[tuple[dict[str, object], list[float], list[float]]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for row in selected:
        history = [float(value) for value in row["history"]]
        step = max(1, len(history) // 240)
        xs = list(range(0, len(history), step))
        ys = [math.log10(history[i] + 1e-9) for i in xs]
        prepared.append((row, xs, ys))
        all_x.extend(xs)
        all_y.extend(ys)

    for index, (row, xs, ys) in enumerate(prepared):
        points = scale_points(xs, ys, box, min(all_x), max(all_x), min(all_y), max(all_y))
        color = colors[index % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        label = f"pop={row['population_size']}, gen={row['generations']}"
        draw.line((810, 145 + index * 34, 850, 145 + index * 34), fill=color, width=4)
        draw.text((862, 134 + index * 34), label, font=font(15), fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def plot_speedup(summaries: list[dict[str, object]], output_path: Path) -> None:
    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    box = (95, 105, 1110, 610)
    draw_axes(draw, box, "Parallel speedup for fitness evaluation", "speedup", "processes")

    processes = [float(row["processes"]) for row in summaries]
    speedups = [float(row["speedup"]) for row in summaries]
    ideal = processes
    x_min, x_max = min(processes), max(processes)
    y_min, y_max = 0.0, max(max(speedups), max(ideal)) * 1.05
    speedup_points = scale_points(processes, speedups, box, x_min, x_max, y_min, y_max)
    ideal_points = scale_points(processes, ideal, box, x_min, x_max, y_min, y_max)
    draw.line(ideal_points, fill=(180, 180, 180), width=2)
    draw.line(speedup_points, fill=(46, 116, 181), width=4)

    for point, row in zip(speedup_points, summaries):
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(46, 116, 181))
        draw.text((x - 16, y - 34), f"{float(row['speedup']):.2f}x", font=font(14), fill=(20, 33, 61))
        draw.text((x - 12, box[3] + 12), str(row["processes"]), font=font(14), fill=(40, 40, 40))

    draw.line((810, 150, 850, 150), fill=(46, 116, 181), width=4)
    draw.text((862, 139), "measured", font=font(15), fill=(40, 40, 40))
    draw.line((810, 184, 850, 184), fill=(180, 180, 180), width=2)
    draw.text((862, 173), "ideal", font=font(15), fill=(40, 40, 40))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def plot_function_quality(rows: list[dict[str, object]], output_path: Path) -> None:
    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(22)
    body_font = font(16)
    draw.text((95, 70), "Best final objective by benchmark function", font=title_font, fill=(20, 33, 61))
    chart_left, chart_top, chart_right, chart_bottom = 130, 150, 1080, 610
    max_log = max(math.log10(float(row["best_value"]) + 1.0) for row in rows)
    bar_width = 170
    gap = 90
    colors = [(46, 116, 181), (31, 78, 121), (122, 90, 0)]
    for index, row in enumerate(rows):
        value = float(row["best_value"])
        log_value = math.log10(value + 1.0)
        height = int(log_value / max_log * (chart_bottom - chart_top)) if max_log > 0 else 1
        x0 = chart_left + index * (bar_width + gap)
        y0 = chart_bottom - height
        draw.rectangle((x0, y0, x0 + bar_width, chart_bottom), fill=colors[index % len(colors)])
        draw.text((x0, chart_bottom + 18), str(row["objective"]), font=body_font, fill=(40, 40, 40))
        draw.text((x0, y0 - 28), f"{value:.3g}", font=body_font, fill=(20, 33, 61))
    draw.line((chart_left - 25, chart_bottom, chart_right, chart_bottom), fill=(45, 55, 72), width=2)
    draw.line((chart_left - 25, chart_top, chart_left - 25, chart_bottom), fill=(45, 55, 72), width=2)
    draw.text((25, chart_top), "log10(best + 1)", font=font(13), fill=(70, 70, 70))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGA coursework experiments.")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--eval-repeats", type=int, default=35)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []

    for population_size in [50, 100, 500]:
        for generations in [100, 500, 1000]:
            config = make_config(
                "rastrigin",
                population_size=population_size,
                generations=generations,
                processes=4,
                seed=200000 + population_size * 10000 + generations,
                eval_repeats=args.eval_repeats,
            )
            row = run_and_label("convergence_grid", config)
            convergence_rows.append(row)
            raw_rows.append(row)

    representative_convergence_rows = [
        row
        for row in convergence_rows
        if (int(row["population_size"]), int(row["generations"])) in PLOT_CONVERGENCE_CONFIGS
    ]

    speed_rows: list[dict[str, object]] = []
    for processes in [1, 2, 4, 8]:
        for replicate in range(args.replicates):
            config = make_config(
                "rastrigin",
                population_size=300,
                generations=400,
                processes=processes,
                seed=5000 + processes * 100 + replicate,
                eval_repeats=args.eval_repeats,
            )
            row = run_and_label("speedup", config)
            speed_rows.append(row)
            raw_rows.append(row)

    function_rows: list[dict[str, object]] = []
    for objective in ["sphere", "rastrigin", "rosenbrock"]:
        config = make_config(
            objective,
            population_size=200,
            generations=500,
            processes=4,
            seed=7000 + len(function_rows),
            eval_repeats=args.eval_repeats,
        )
        row = run_and_label("function_quality", config)
        function_rows.append(row)
        raw_rows.append(row)

    serializable_rows = [{key: value for key, value in row.items() if key != "history"} for row in raw_rows]
    write_csv(
        RESULTS_DIR / "experiment_runs.csv",
        serializable_rows,
        [
            "label",
            "objective",
            "dimensions",
            "population_size",
            "generations",
            "processes",
            "eval_repeats",
            "seed",
            "best_value",
            "final_best_value",
            "runtime_sec",
            "avg_generation_ms",
            "wall_clock_sec",
        ],
    )

    with (RESULTS_DIR / "convergence_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "objective", "population_size", "generations", "processes", "generation", "best_value"])
        for row in raw_rows:
            for generation, value in enumerate(row["history"]):
                writer.writerow(
                    [
                        row["label"],
                        row["objective"],
                        row["population_size"],
                        row["generations"],
                        row["processes"],
                        generation,
                        f"{float(value):.12g}",
                    ]
                )

    speed_summaries = summarize_by_process(speed_rows)
    write_csv(
        RESULTS_DIR / "speedup_summary.csv",
        speed_summaries,
        [
            "processes",
            "runs",
            "mean_runtime_sec",
            "stdev_runtime_sec",
            "mean_best_value",
            "speedup",
            "efficiency_pct",
        ],
    )

    plot_convergence(representative_convergence_rows, PLOTS_DIR / "convergence.png")
    plot_speedup(speed_summaries, PLOTS_DIR / "speedup.png")
    plot_function_quality(function_rows, PLOTS_DIR / "function_quality.png")

    print(f"Done. Results written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
