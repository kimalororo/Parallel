from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

from variant5_gpu import (
    KernelConfig,
    collect_input_videos,
    explain_error,
    processor_for_mode,
    read_batch,
)


MODES = ("cuda2d", "cuda3d", "opencl1d", "opencl2d")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def quant_count(value: str) -> int:
    parsed = int(value)
    if parsed < 4 or parsed > 10:
        raise argparse.ArgumentTypeError("n must be in range 4..10")
    return parsed


def parse_csv_ints(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        result.append(positive_int(part))
    if not result:
        raise argparse.ArgumentTypeError("provide at least one integer value")
    return result


def parse_2d_sizes(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise argparse.ArgumentTypeError("2D sizes must look like 8x8,16x16")
        x_text, y_text = part.split("x", 1)
        result.append((positive_int(x_text), positive_int(y_text)))
    if not result:
        raise argparse.ArgumentTypeError("provide at least one 2D size")
    return result


def parse_modes(value: str) -> list[str]:
    if value == "all":
        return list(MODES)

    modes = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes: {', '.join(unknown)}")
    if not modes:
        raise argparse.ArgumentTypeError("provide at least one mode")
    return modes


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark lab 2 variant 5 GPU kernels and generate CSV/PNG results.",
    )
    parser.add_argument("-i", "--input", default=Path("input"), type=Path)
    parser.add_argument("-o", "--output", default=Path("benchmark"), type=Path)
    parser.add_argument("--n", default=6, type=quant_count)
    parser.add_argument("--batch", default=16, type=positive_int)
    parser.add_argument("--repeats", default=3, type=positive_int)
    parser.add_argument("--modes", default="all", type=parse_modes)
    parser.add_argument("--sizes-1d", default=parse_csv_ints("64,128,256,512"), type=parse_csv_ints)
    parser.add_argument("--sizes-2d", default=parse_2d_sizes("8x8,16x16,32x16"), type=parse_2d_sizes)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="build PNG graphs from benchmark_summary.csv without running benchmarks",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        type=Path,
        help="summary CSV for --plot-only, default: <output>/benchmark_summary.csv",
    )
    parser.add_argument(
        "--opencl-device",
        default="gpu",
        choices=["gpu", "cpu", "any"],
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="disable one untimed warmup batch before measured repeats",
    )
    return parser.parse_args(argv)


def mode_configs(mode: str, args: argparse.Namespace) -> list[tuple[str, int, KernelConfig]]:
    base = KernelConfig(opencl_device=args.opencl_device)

    if mode == "opencl1d":
        return [
            (
                f"local={local}",
                local,
                replace(base, opencl_local_1d=local),
            )
            for local in args.sizes_1d
        ]

    return [
        (
            f"{x}x{y}",
            x * y,
            replace(
                base,
                threads_x=x,
                threads_y=y,
                opencl_local_x=x,
                opencl_local_y=y,
            ),
        )
        for x, y in args.sizes_2d
    ]


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Cannot detect frame size: {video_path}")

    return capture, width, height


def warmup(video_path: Path, mode: str, config: KernelConfig, n: int, batch_size: int) -> None:
    processor = processor_for_mode(mode)
    capture, width, height = open_video(video_path)
    try:
        batch = read_batch(capture, min(batch_size, 2), width, height)
        if batch.shape[0] > 0:
            processor(batch, n, config)
    finally:
        capture.release()


def benchmark_once(video_path: Path, mode: str, config: KernelConfig, n: int, batch_size: int) -> tuple[int, float]:
    processor = processor_for_mode(mode)
    capture, width, height = open_video(video_path)
    frames = 0
    gpu_ms = 0.0

    try:
        while True:
            batch = read_batch(capture, batch_size, width, height)
            if batch.shape[0] == 0:
                break

            started = time.perf_counter()
            processor(batch, n, config)
            gpu_ms += (time.perf_counter() - started) * 1000.0
            frames += int(batch.shape[0])
    finally:
        capture.release()

    return frames, gpu_ms


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_summary_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = str(row["video"]), str(row["mode"]), str(row["config"])
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (video, mode, config), values in sorted(grouped.items()):
        times = [float(row["gpu_ms"]) for row in values]
        frames = int(values[0]["frames"])
        thread_count = int(values[0]["thread_count"])
        avg_ms = statistics.mean(times)
        summary.append(
            {
                "video": video,
                "mode": mode,
                "config": config,
                "thread_count": thread_count,
                "repeats": len(times),
                "frames": frames,
                "avg_gpu_ms": f"{avg_ms:.3f}",
                "min_gpu_ms": f"{min(times):.3f}",
                "max_gpu_ms": f"{max(times):.3f}",
                "avg_ms_per_frame": f"{avg_ms / frames:.6f}" if frames else "",
            }
        )
    return summary


def plot_summary(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, PNG graphs were not generated.", file=sys.stderr)
        return

    videos = sorted({str(row["video"]) for row in summary_rows})
    for video in videos:
        video_rows = [row for row in summary_rows if row["video"] == video]
        modes = sorted({str(row["mode"]) for row in video_rows})

        plt.figure(figsize=(9, 5))
        for mode in modes:
            mode_rows = sorted(
                [row for row in video_rows if row["mode"] == mode],
                key=lambda row: int(row["thread_count"]),
            )
            x_values = [int(row["thread_count"]) for row in mode_rows]
            y_values = [float(row["avg_gpu_ms"]) for row in mode_rows]
            labels = [str(row["config"]) for row in mode_rows]
            plt.plot(x_values, y_values, marker="o", label=mode)
            for x_value, y_value, label in zip(x_values, y_values, labels):
                plt.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(0, 6), ha="center")

        plt.title(f"Среднее время GPU-обработки: {video}")
        plt.xlabel("Количество потоков / work-items в блоке или группе")
        plt.ylabel("Среднее время, мс")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{Path(video).stem}_avg_time.png", dpi=160)
        plt.close()


def run(args: argparse.Namespace) -> None:
    if args.plot_only:
        args.output.mkdir(parents=True, exist_ok=True)
        summary_path = args.summary_csv or args.output / "benchmark_summary.csv"
        if not summary_path.exists():
            raise RuntimeError(f"Summary CSV not found: {summary_path}")

        summary_rows = read_summary_csv(summary_path)
        if not summary_rows:
            raise RuntimeError(f"Summary CSV is empty: {summary_path}")

        plot_summary(summary_rows, args.output)
        print(f"Graphs: {args.output / '*_avg_time.png'}")
        return

    videos = collect_input_videos(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for video in videos:
        for mode in args.modes:
            for config_label, thread_count, config in mode_configs(mode, args):
                if not args.no_warmup:
                    try:
                        warmup(video, mode, config, args.n, args.batch)
                    except Exception as exc:
                        print(f"Warmup skipped for {video.name} / {mode} / {config_label}: {explain_error(exc)}")

                for repeat in range(1, args.repeats + 1):
                    row = {
                        "video": video.name,
                        "mode": mode,
                        "config": config_label,
                        "thread_count": thread_count,
                        "repeat": repeat,
                        "frames": "",
                        "gpu_ms": "",
                        "status": "ok",
                        "error": "",
                    }

                    try:
                        frames, gpu_ms = benchmark_once(video, mode, config, args.n, args.batch)
                        row["frames"] = frames
                        row["gpu_ms"] = f"{gpu_ms:.3f}"
                        print(
                            f"{video.name} | {mode:8} | {config_label:9} | "
                            f"run {repeat}: {gpu_ms:.3f} ms"
                        )
                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = explain_error(exc)
                        print(
                            f"{video.name} | {mode:8} | {config_label:9} | "
                            f"run {repeat}: skipped - {row['error']}",
                            file=sys.stderr,
                        )

                    rows.append(row)

    raw_fields = [
        "video",
        "mode",
        "config",
        "thread_count",
        "repeat",
        "frames",
        "gpu_ms",
        "status",
        "error",
    ]
    summary_fields = [
        "video",
        "mode",
        "config",
        "thread_count",
        "repeats",
        "frames",
        "avg_gpu_ms",
        "min_gpu_ms",
        "max_gpu_ms",
        "avg_ms_per_frame",
    ]

    summary_rows = summarize(rows)
    write_csv(args.output / "benchmark_raw.csv", rows, raw_fields)
    write_csv(args.output / "benchmark_summary.csv", summary_rows, summary_fields)
    plot_summary(summary_rows, args.output)

    print(f"\nRaw results: {args.output / 'benchmark_raw.csv'}")
    print(f"Summary: {args.output / 'benchmark_summary.csv'}")
    print(f"Graphs: {args.output / '*_avg_time.png'}")


def main(argv: list[str]) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
