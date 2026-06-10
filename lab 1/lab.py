import os
import cv2
import time
import math
import platform
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor

try:
    import psutil
except ImportError:
    psutil = None


def chunkify(lst, n_chunks):
    """
    Разбить список на n_chunks примерно равных частей.
    """
    if n_chunks <= 0:
        return [lst]

    chunk_size = math.ceil(len(lst) / n_chunks)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def read_video_to_frames(video_path):
    """
    Считать видеофайл целиком в массив кадров.
    Возвращает:
        frames - список кадров
        fps - частота кадров
        width - ширина кадра
        height - высота кадра
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"Видео не содержит кадров: {video_path}")

    return frames, fps, width, height


def process_frame(frame):
    """
    Обработка одного кадра.

    Шаги:
    1. Вычисление матрицы интенсивности:
       I = (R + G + B) / 3
    2. Выделение пикселей первого кванта интенсивности:
       I < 64
    3. Построение бинарной маски.
    4. Очистка шумов морфологической операцией.
    5. Поиск контуров выделенных областей.
    6. Нанесение красной границы на исходный кадр.
    """
    # Средняя интенсивность по трем каналам
    intensity = frame.mean(axis=2).astype("uint8")

    # Пиксели первого кванта интенсивности
    mask = (intensity < 64).astype("uint8") * 255

    # Очистка мелких шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Копия исходного кадра
    result = frame.copy()

    # Красный цвет в OpenCV задается в формате BGR
    cv2.drawContours(result, contours, -1, (0, 0, 255), thickness=1)

    return result


def process_chunk(frames_chunk):
    """
    Обработка блока кадров в одном потоке.
    """
    return [process_frame(frame) for frame in frames_chunk]


def parallel_process_frames(frames, num_workers):
    """
    Параллельная обработка кадров с использованием потоков.

    Для уменьшения накладных расходов массив кадров разбивается
    на блоки, и каждый блок обрабатывается отдельным потоком.
    """
    if num_workers <= 1:
        return [process_frame(frame) for frame in frames]

    chunks = chunkify(frames, num_workers)

    processed_frames = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_chunk, chunks)
        for chunk_result in results:
            processed_frames.extend(chunk_result)

    return processed_frames


def save_video(frames, output_path, fps, width, height):
    """
    Сохранить массив кадров в видеофайл.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Не удалось создать выходной видеофайл: {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()


def save_preview_frames(frames, output_dir, count=5):
    """
    Сохранить несколько кадров в формате PNG для отчета.
    """
    os.makedirs(output_dir, exist_ok=True)

    step = max(1, len(frames) // count)
    idx = 0
    saved = 0

    while idx < len(frames) and saved < count:
        filename = os.path.join(output_dir, f"frame_{saved + 1:02d}.png")
        cv2.imwrite(filename, frames[idx])
        idx += step
        saved += 1


def benchmark(video_path, workers_list, repeats, output_dir):
    """
    Выполнить замеры времени обработки одного видео
    для разных чисел потоков.
    """
    frames, fps, width, height = read_video_to_frames(video_path)
    results = []

    for workers in workers_list:
        times = []

        for run_idx in range(repeats):
            start = time.perf_counter()
            processed_frames = parallel_process_frames(frames, workers)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Сохраняем выходное видео только для первого прогона
            if run_idx == 0:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                out_name = f"{base_name}_threads_{workers}.mp4"
                out_path = os.path.join(output_dir, out_name)
                save_video(processed_frames, out_path, fps, width, height)

        avg_time = statistics.mean(times)

        results.append({
            "video": os.path.basename(video_path),
            "workers": workers,
            "run1": times[0] if len(times) > 0 else None,
            "run2": times[1] if len(times) > 1 else None,
            "run3": times[2] if len(times) > 2 else None,
            "avg": avg_time
        })

    return results


def print_results_table(results):
    """
    Вывести таблицу результатов в консоль.
    """
    print("\nРезультаты тестирования:")
    print("-" * 78)
    print(f"{'Видео':20} {'Потоки':>8} {'Запуск 1':>12} {'Запуск 2':>12} {'Запуск 3':>12} {'Среднее':>12}")
    print("-" * 78)

    for row in results:
        r1 = f"{row['run1']:.4f}" if row["run1"] is not None else "-"
        r2 = f"{row['run2']:.4f}" if row["run2"] is not None else "-"
        r3 = f"{row['run3']:.4f}" if row["run3"] is not None else "-"
        avg = f"{row['avg']:.4f}" if row["avg"] is not None else "-"

        print(f"{row['video'][:20]:20} {row['workers']:>8} {r1:>12} {r2:>12} {r3:>12} {avg:>12}")


def save_results_csv(results, csv_path):
    """
    Сохранить результаты тестирования в CSV-файл.
    """
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("video,threads,run1,run2,run3,avg\n")

        for row in results:
            run1 = f"{row['run1']:.6f}" if row["run1"] is not None else ""
            run2 = f"{row['run2']:.6f}" if row["run2"] is not None else ""
            run3 = f"{row['run3']:.6f}" if row["run3"] is not None else ""
            avg = f"{row['avg']:.6f}" if row["avg"] is not None else ""

            f.write(
                f"{row['video']},{row['workers']},{run1},{run2},{run3},{avg}\n"
            )


def hardware_info():
    """
    Получить сведения об аппаратной базе.
    """
    info = {
        "OS": platform.platform(),
        "CPU": platform.processor(),
        "CPU cores (logical)": os.cpu_count()
    }

    if psutil is not None:
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        info["RAM (GB)"] = round(mem_gb, 2)
    else:
        info["RAM (GB)"] = "psutil не установлен"

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №1. Многопоточная обработка видео на CPU. Вариант 2."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Список входных видеофайлов"
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Число потоков"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Количество повторов для замеров"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Папка для результатов"
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Сохранить несколько кадров для отчета"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Аппаратная база:")
    info = hardware_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    all_results = []

    for video_path in args.videos:
        print(f"\nОбработка видео: {video_path}")

        video_results = benchmark(
            video_path=video_path,
            workers_list=args.workers,
            repeats=args.repeats,
            output_dir=args.output
        )
        all_results.extend(video_results)

        if args.save_preview:
            frames, _, _, _ = read_video_to_frames(video_path)
            processed_frames = parallel_process_frames(frames, args.workers[0])

            preview_dir = os.path.join(
                args.output,
                f"preview_{os.path.splitext(os.path.basename(video_path))[0]}"
            )
            save_preview_frames(processed_frames, preview_dir)

    print_results_table(all_results)

    csv_path = os.path.join(args.output, "benchmark_results.csv")
    save_results_csv(all_results, csv_path)

    print(f"\nГотово. Результаты сохранены в папке: {args.output}")
    print(f"CSV с результатами сохранён: {csv_path}")


if __name__ == "__main__":
    main()