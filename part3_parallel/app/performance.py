"""Модуль замера производительности: последовательная vs параллельная обработка."""

import json
from pathlib import Path
import sys
import time
from typing import List, Tuple
import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from part3_parallel.app.augment import augment_single_image  # noqa: E402
from part3_parallel.app.producer_consumer import run_parallel_pipeline  # noqa: E402


def run_sequential_pipeline(data_dir: Path, target_size: Tuple[int, int] = (64, 64)) -> Tuple[int, int]:
    """Последовательное выполнение: чтение в одном потоке и аугментация в одном процессе."""
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]

    augmented_items: List[np.ndarray] = []
    for file_path in files:
        with Image.open(file_path) as img:
            arr = np.asarray(img.convert("RGB").resize(target_size), dtype=np.float32) / 255.0
            results = augment_single_image(arr)
            augmented_items.extend(results)

    return len(files), len(augmented_items)


def main() -> None:
    data_dir = root_dir / "data" / "input" / "images"
    report_path = root_dir / "performance_report.json"

    print("=" * 60)
    print("      ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ (ЧАСТЬ 3)")
    print("=" * 60)

    # 1. Замер последовательного алгоритма
    print("\n[1/2] Запуск последовательного выполнения...")
    t0 = time.perf_counter()
    seq_files, seq_augmented = run_sequential_pipeline(data_dir)
    seq_time = time.perf_counter() - t0
    print(f"Последовательно: {seq_files} файлов -> {seq_augmented} аугментаций за {seq_time:.4f} сек.")

    # 2. Замер параллельного алгоритма
    print("\n[2/2] Запуск параллельного конвейера (Threads + Processes)...")
    t1 = time.perf_counter()
    par_files, par_augmented = run_parallel_pipeline(data_dir, num_consumers=4, num_processes=4)
    par_time = time.perf_counter() - t1
    print(f"Параллельно:     {par_files} файлов -> {par_augmented} аугментаций за {par_time:.4f} сек.")

    # Расчет ускорения
    speedup = round(seq_time / par_time, 2) if par_time > 0 else 1.0

    report_data = {
        "files_processed": seq_files,
        "total_augmented_samples": seq_augmented,
        "sequential_time_sec": round(seq_time, 4),
        "parallel_time_sec": round(par_time, 4),
        "speedup_ratio": f"{speedup}x",
        "parameters": {
            "num_io_threads": 4,
            "num_cpu_processes": 4,
            "augmentations_per_image": 4,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"ИТОГОВОЕ УСКОРЕНИЕ: {speedup}x")
    print(f"Отчет сохранен в:   {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()