from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    from numba import cuda as NUMBA_CUDA
except ImportError:
    NUMBA_CUDA = None


Mode = str
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DLL_DIRECTORY_HANDLES = []


@dataclass(frozen=True)
class KernelConfig:
    threads_x: int = 16
    threads_y: int = 16
    opencl_local_1d: int = 256
    opencl_local_x: int = 16
    opencl_local_y: int = 16
    opencl_device: str = "gpu"


@dataclass(frozen=True)
class Options:
    input_path: Path
    output_path: Path
    mode: Mode
    n: int
    batch_size: int
    kernel: KernelConfig


OPENCL_SOURCE = r"""
inline uchar quantize_channel(uchar value, int n) {
    int bin = ((int)value * n) / 256;
    if (bin >= n) {
        bin = n - 1;
    }

    int center = ((2 * bin + 1) * 256 + n) / (2 * n);
    if (center > 255) {
        center = 255;
    }
    return (uchar)center;
}

__kernel void quantize_1d(__global uchar* data, int total_values, int n) {
    int index = get_global_id(0);
    if (index >= total_values) {
        return;
    }

    data[index] = quantize_channel(data[index], n);
}

__kernel void quantize_2d(__global uchar* frames,
                          int width,
                          int height,
                          int channels,
                          int frame_count,
                          int n) {
    int x = get_global_id(0);
    int global_y = get_global_id(1);
    int frame_index = global_y / height;
    int y = global_y - frame_index * height;

    if (x >= width || y >= height || frame_index >= frame_count) {
        return;
    }

    int index = ((frame_index * height + y) * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
        frames[index + c] = quantize_channel(frames[index + c], n);
    }
}
"""

OPENCL_CACHE = {}


if NUMBA_CUDA is not None:

    @NUMBA_CUDA.jit(device=True)
    def quantize_channel_cuda(value, n):
        bin_index = (int(value) * n) // 256
        if bin_index >= n:
            bin_index = n - 1

        center = ((2 * bin_index + 1) * 256 + n) // (2 * n)
        if center > 255:
            center = 255
        return center

    @NUMBA_CUDA.jit
    def quantize_cuda_2d_kernel(frames, width, height, channels, frame_index, n):
        x = NUMBA_CUDA.blockIdx.x * NUMBA_CUDA.blockDim.x + NUMBA_CUDA.threadIdx.x
        y = NUMBA_CUDA.blockIdx.y * NUMBA_CUDA.blockDim.y + NUMBA_CUDA.threadIdx.y

        if x >= width or y >= height:
            return

        for c in range(channels):
            frames[frame_index, y, x, c] = quantize_channel_cuda(frames[frame_index, y, x, c], n)

    @NUMBA_CUDA.jit
    def quantize_cuda_3d_kernel(frames, width, height, channels, frame_count, n):
        x = NUMBA_CUDA.blockIdx.x * NUMBA_CUDA.blockDim.x + NUMBA_CUDA.threadIdx.x
        y = NUMBA_CUDA.blockIdx.y * NUMBA_CUDA.blockDim.y + NUMBA_CUDA.threadIdx.y
        z = NUMBA_CUDA.blockIdx.z

        if x >= width or y >= height or z >= frame_count:
            return

        for c in range(channels):
            frames[z, y, x, c] = quantize_channel_cuda(frames[z, y, x, c], n)


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


def parse_args(argv: list[str]) -> Options:
    parser = argparse.ArgumentParser(
        description="Lab 2 variant 5: GPU video color quantization with CUDA and OpenCL.",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=Path("input"),
        type=Path,
        help="input video or directory with videos, default: input",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("output"),
        type=Path,
        help="output video or output directory, default: output",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="all",
        choices=["cuda2d", "cuda3d", "opencl1d", "opencl2d", "all"],
        help="kernel variant",
    )
    parser.add_argument("--n", default=6, type=quant_count, help="quant count, 4..10")
    parser.add_argument("--batch", default=16, type=positive_int, help="frames per GPU batch")
    parser.add_argument("--threads-x", default=16, type=positive_int, help="CUDA blockDim.x")
    parser.add_argument("--threads-y", default=16, type=positive_int, help="CUDA blockDim.y")
    parser.add_argument("--local-1d", default=256, type=positive_int, help="OpenCL 1D local size")
    parser.add_argument("--local-x", default=None, type=positive_int, help="OpenCL local size X")
    parser.add_argument("--local-y", default=None, type=positive_int, help="OpenCL local size Y")
    parser.add_argument(
        "--opencl-device",
        default="gpu",
        choices=["gpu", "cpu", "any"],
        help="OpenCL device type",
    )
    args = parser.parse_args(argv)

    local_x = args.local_x if args.local_x is not None else args.threads_x
    local_y = args.local_y if args.local_y is not None else args.threads_y

    return Options(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        n=args.n,
        batch_size=args.batch,
        kernel=KernelConfig(
            threads_x=args.threads_x,
            threads_y=args.threads_y,
            opencl_local_1d=args.local_1d,
            opencl_local_x=local_x,
            opencl_local_y=local_y,
            opencl_device=args.opencl_device,
        ),
    )


def mode_output_path(base_path: Path, mode: Mode) -> Path:
    suffix = base_path.suffix or ".mp4"
    return base_path.with_name(f"{base_path.stem}_{mode}{suffix}")


def collect_input_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported input video extension: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        videos = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
            raise RuntimeError(f"No video files found in {input_path}. Supported: {extensions}")
        return videos

    raise RuntimeError(f"Input path does not exist: {input_path}")


def is_directory_output(output_path: Path, many_inputs: bool) -> bool:
    return many_inputs or output_path.is_dir() or output_path.suffix == ""


def output_path_for_video(
    output_path: Path,
    input_video: Path,
    mode: Mode,
    many_inputs: bool,
    all_modes: bool,
) -> Path:
    if is_directory_output(output_path, many_inputs):
        return output_path / f"{input_video.stem}_{mode}.mp4"

    if all_modes:
        return mode_output_path(output_path, mode)

    return output_path


def round_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def cuda_toolkit_homes() -> list[Path]:
    homes: list[Path] = []

    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            homes.append(Path(value))

    for name, value in os.environ.items():
        if name.startswith("CUDA_PATH_V") and value:
            homes.append(Path(value))

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        homes.append(Path(nvcc_path).resolve().parents[1])

    unique_homes: list[Path] = []
    seen: set[str] = set()
    for home in homes:
        try:
            key = str(home.resolve()).lower()
        except OSError:
            key = str(home).lower()

        if key not in seen:
            seen.add(key)
            unique_homes.append(home)

    return unique_homes


def find_nvvm_dll() -> Path | None:
    for home in cuda_toolkit_homes():
        candidates = [
            *home.glob("nvvm/bin/x64/nvvm*.dll"),
            *home.glob("nvvm/bin/nvvm*.dll"),
        ]
        if candidates:
            return candidates[0]
    return None


def find_cuda_runtime_dir() -> Path | None:
    for home in cuda_toolkit_homes():
        for directory in (home / "bin" / "x64", home / "bin"):
            if list(directory.glob("cudart*.dll")):
                return directory
    return None


def configure_numba_nvvm() -> None:
    try:
        from numba.cuda import cuda_paths
    except ImportError:
        return

    paths = dict(cuda_paths.get_cuda_paths())
    current_nvvm = paths.get("nvvm")
    if current_nvvm is not None and current_nvvm.info and Path(current_nvvm.info).exists():
        return

    nvvm_dll = find_nvvm_dll()
    if nvvm_dll is None:
        raise RuntimeError(
            "CUDA mode cannot find NVVM. Install NVIDIA CUDA Toolkit and check `nvcc --version`."
        )

    runtime_dir = find_cuda_runtime_dir()

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(nvvm_dll.parent)))
        if runtime_dir is not None:
            DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_dir)))

    os.environ.setdefault("NUMBAPRO_NVVM", str(nvvm_dll))
    paths["nvvm"] = cuda_paths._env_path_tuple(
        by="CUDA Toolkit auto-detect",
        info=str(nvvm_dll),
    )
    if runtime_dir is not None:
        paths["cudalib_dir"] = cuda_paths._env_path_tuple(
            by="CUDA Toolkit auto-detect",
            info=str(runtime_dir),
        )
    cuda_paths.get_cuda_paths._cached_result = paths


def load_cuda_kernels():
    if NUMBA_CUDA is None:
        raise RuntimeError(
            "CUDA mode requires numba. Install it with: pip install numba"
        )

    configure_numba_nvvm()
    return NUMBA_CUDA, quantize_cuda_2d_kernel, quantize_cuda_3d_kernel


def process_cuda_2d(batch: np.ndarray, n: int, config: KernelConfig) -> np.ndarray:
    cuda, quantize_cuda_2d, _ = load_cuda_kernels()
    frame_count, height, width, channels = batch.shape
    device_batch = cuda.to_device(np.ascontiguousarray(batch))

    block = (config.threads_x, config.threads_y)
    grid = (round_up(width, block[0]) // block[0], round_up(height, block[1]) // block[1])

    for frame_index in range(frame_count):
        quantize_cuda_2d[grid, block](device_batch, width, height, channels, frame_index, n)

    cuda.synchronize()
    return device_batch.copy_to_host()


def process_cuda_3d(batch: np.ndarray, n: int, config: KernelConfig) -> np.ndarray:
    cuda, _, quantize_cuda_3d = load_cuda_kernels()
    frame_count, height, width, channels = batch.shape
    device_batch = cuda.to_device(np.ascontiguousarray(batch))

    block = (config.threads_x, config.threads_y)
    grid = (
        round_up(width, block[0]) // block[0],
        round_up(height, block[1]) // block[1],
        frame_count,
    )

    quantize_cuda_3d[grid, block](device_batch, width, height, channels, frame_count, n)
    cuda.synchronize()
    return device_batch.copy_to_host()


def select_opencl_device(cl, device_kind: str):
    preferred_types = {
        "gpu": [cl.device_type.GPU],
        "cpu": [cl.device_type.CPU],
        "any": [cl.device_type.GPU, cl.device_type.ACCELERATOR, cl.device_type.CPU],
    }[device_kind]

    for device_type in preferred_types:
        for platform in cl.get_platforms():
            try:
                devices = platform.get_devices(device_type=device_type)
            except cl.LogicError:
                devices = []
            if devices:
                return platform, devices[0]

    raise RuntimeError(f"No OpenCL {device_kind} device found")


def build_opencl_program(config: KernelConfig):
    try:
        import pyopencl as cl
    except ImportError as exc:
        raise RuntimeError(
            "OpenCL mode requires pyopencl. Install it with: pip install pyopencl"
        ) from exc

    cache_key = config.opencl_device
    if cache_key in OPENCL_CACHE:
        return OPENCL_CACHE[cache_key]

    _, device = select_opencl_device(cl, config.opencl_device)
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, OPENCL_SOURCE).build()
    kernels = {
        "quantize_1d": cl.Kernel(program, "quantize_1d"),
        "quantize_2d": cl.Kernel(program, "quantize_2d"),
    }
    value = cl, context, queue, program, kernels, device
    OPENCL_CACHE[cache_key] = value
    return value


def process_opencl_1d(batch: np.ndarray, n: int, config: KernelConfig) -> np.ndarray:
    cl, context, queue, _, kernels, _ = build_opencl_program(config)
    flat = np.ascontiguousarray(batch).reshape(-1)

    buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat)
    total_values = np.int32(flat.size)
    local = (config.opencl_local_1d,)
    global_size = (round_up(flat.size, local[0]),)

    kernels["quantize_1d"](queue, global_size, local, buffer, total_values, np.int32(n))
    result = np.empty_like(flat)
    cl.enqueue_copy(queue, result, buffer)
    queue.finish()
    return result.reshape(batch.shape)


def process_opencl_2d(batch: np.ndarray, n: int, config: KernelConfig) -> np.ndarray:
    cl, context, queue, _, kernels, _ = build_opencl_program(config)
    frame_count, height, width, channels = batch.shape
    flat = np.ascontiguousarray(batch).reshape(-1)

    buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat)
    local = (config.opencl_local_x, config.opencl_local_y)
    global_size = (round_up(width, local[0]), round_up(height * frame_count, local[1]))

    kernels["quantize_2d"](
        queue,
        global_size,
        local,
        buffer,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        np.int32(frame_count),
        np.int32(n),
    )
    result = np.empty_like(flat)
    cl.enqueue_copy(queue, result, buffer)
    queue.finish()
    return result.reshape(batch.shape)


def processor_for_mode(mode: Mode) -> Callable[[np.ndarray, int, KernelConfig], np.ndarray]:
    processors = {
        "cuda2d": process_cuda_2d,
        "cuda3d": process_cuda_3d,
        "opencl1d": process_opencl_1d,
        "opencl2d": process_opencl_2d,
    }
    return processors[mode]


def read_batch(capture: cv2.VideoCapture, batch_size: int, width: int, height: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    for _ in range(batch_size):
        ok, frame = capture.read()
        if not ok:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            raise RuntimeError("Video frame size changed while reading")

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError("Unsupported video frame channel count")

        frames.append(np.ascontiguousarray(frame, dtype=np.uint8))

    if not frames:
        return np.empty((0, height, width, 3), dtype=np.uint8)

    return np.stack(frames, axis=0)


def write_batch(writer: cv2.VideoWriter, batch: np.ndarray) -> None:
    for frame in batch:
        writer.write(frame)


def process_video(options: Options, input_path: Path, mode: Mode, output_path: Path) -> None:
    processor = processor_for_mode(mode)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        raise RuntimeError("Cannot detect input video frame size")
    if fps <= 0:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {output_path}")

    total_frames = 0
    processing_ms = 0.0

    try:
        while True:
            batch = read_batch(capture, options.batch_size, width, height)
            if batch.shape[0] == 0:
                break

            started = time.perf_counter()
            processed = processor(batch, options.n, options.kernel)
            processing_ms += (time.perf_counter() - started) * 1000.0

            write_batch(writer, processed)
            total_frames += int(processed.shape[0])
    finally:
        writer.release()
        capture.release()

    print(
        f"Mode {mode}: wrote {total_frames} frames to {output_path}, "
        f"GPU stage {processing_ms:.3f} ms"
    )


def explain_error(error: Exception) -> str:
    message = str(error)
    if "libNVVM cannot be found" in message or "nvvm.dll" in message:
        return (
            "CUDA mode cannot start because nvvm.dll was not found. "
            "Install NVIDIA CUDA Toolkit, reopen PowerShell, and check `nvcc --version`."
        )
    if "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" in message or "Unsupported .version" in message:
        return (
            "CUDA mode cannot start because the NVIDIA driver is older than the CUDA Toolkit "
            "used to generate PTX. Update the NVIDIA driver or install a CUDA Toolkit version "
            "that matches the CUDA version shown by `nvidia-smi`."
        )
    return message


def run(options: Options) -> None:
    modes = ["cuda2d", "cuda3d", "opencl1d", "opencl2d"] if options.mode == "all" else [options.mode]
    input_videos = collect_input_videos(options.input_path)
    many_inputs = len(input_videos) > 1 or options.input_path.is_dir()
    all_modes = options.mode == "all"
    failures: list[str] = []
    completed = 0

    for input_video in input_videos:
        for mode in modes:
            output_path = output_path_for_video(
                options.output_path,
                input_video,
                mode,
                many_inputs,
                all_modes,
            )
            try:
                process_video(options, input_video, mode, output_path)
                completed += 1
            except Exception as exc:
                if not all_modes:
                    raise

                failure = f"{input_video.name} / {mode}: {explain_error(exc)}"
                failures.append(failure)
                print(f"Skipped {failure}", file=sys.stderr)

    if failures:
        print("\nCompleted modes:", completed)
        print("Skipped modes:")
        for failure in failures:
            print(f"  - {failure}")

    if completed == 0 and failures:
        raise RuntimeError("No modes completed successfully.")


def main(argv: list[str]) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
