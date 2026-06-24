"""Regenerate plots from existing experiment CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from run_experiments import PLOT_CONVERGENCE_CONFIGS, plot_convergence, plot_function_quality, plot_speedup


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def read_dicts(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_convergence_rows() -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], dict[str, object]] = {}
    with (RESULTS_DIR / "convergence_history.csv").open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["label"] != "convergence_grid":
                continue
            key = (row["objective"], row["population_size"], row["generations"], row["processes"])
            group = groups.setdefault(
                key,
                {
                    "objective": row["objective"],
                    "population_size": int(row["population_size"]),
                    "generations": int(row["generations"]),
                    "processes": int(row["processes"]),
                    "history": [],
                },
            )
            group["history"].append(float(row["best_value"]))
    return list(groups.values())


def main() -> None:
    speed_rows = read_dicts(RESULTS_DIR / "speedup_summary.csv")
    function_rows = [row for row in read_dicts(RESULTS_DIR / "experiment_runs.csv") if row["label"] == "function_quality"]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    convergence_rows = [
        row
        for row in load_convergence_rows()
        if (int(row["population_size"]), int(row["generations"])) in PLOT_CONVERGENCE_CONFIGS
    ]
    plot_convergence(convergence_rows, PLOTS_DIR / "convergence.png")
    plot_speedup(speed_rows, PLOTS_DIR / "speedup.png")
    plot_function_quality(function_rows, PLOTS_DIR / "function_quality.png")
    print(f"Plots regenerated in {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
