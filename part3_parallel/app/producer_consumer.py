"""Модуль многопоточной загрузки и многопроцессорной аугментации."""

import multiprocessing as mp
from pathlib import Path
import queue
import threading
from typing import List, Tuple
import numpy as np
from PIL import Image

from part3_parallel.app.augment import augment_single_image

# Маркер завершения очереди (Poison Pill)
SENTINEL = None


class PipelineStats:
    """Потокобезопасный счетчик обработанных файлов."""

    def __init__(self) -> None:
        self.count: int = 0
        self.lock = threading.Lock()

    def increment(self) -> None:
        """Инкремент счетчика в критической секции."""
        with self.lock:
            self.count += 1

    def get_count(self) -> int:
        """Получение текущего значения счетчика."""
        with self.lock:
            return self.count


def producer_worker(directory: Path, file_queue: queue.Queue, num_consumers: int) -> None:
    """Поток-производитель: сканирует каталог и наполняет очередь путями к файлам."""
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            file_queue.put(path)

    # Сигнал завершения для каждого потока-потребителя
    for _ in range(num_consumers):
        file_queue.put(SENTINEL)


def consumer_worker(
    file_queue: queue.Queue,
    image_queue: queue.Queue,
    stats: PipelineStats,
    target_size: Tuple[int, int] = (64, 64),
) -> None:
    """Поток-потребитель: читает файл с диска и кладет массив в следующую очередь."""
    while True:
        file_path = file_queue.get()
        if file_path is SENTINEL:
            file_queue.task_done()
            break

        try:
            with Image.open(file_path) as img:
                arr = np.asarray(img.convert("RGB").resize(target_size), dtype=np.float32) / 255.0
                image_queue.put(arr)
                stats.increment()
        except Exception as err:
            print(f"[Consumer Error] Ошибка чтения {file_path}: {err}")
        finally:
            file_queue.task_done()


def run_parallel_pipeline(
    data_dir: Path,
    num_consumers: int = 4,
    num_processes: int = 4,
) -> Tuple[int, int]:
    """Запуск параллельного конвейера обработки данных.

    :return: Кортеж (количество прочитанных файлов, количество сгенерированных аугментаций).
    """
    file_queue: queue.Queue = queue.Queue()
    image_queue: queue.Queue = queue.Queue()
    stats = PipelineStats()

    # 1. Поток-производитель
    prod_thread = threading.Thread(
        target=producer_worker,
        args=(data_dir, file_queue, num_consumers),
        name="Producer-1",
    )
    prod_thread.start()

    # 2. Потоки-потребители (чтение I/O)
    consumers: List[threading.Thread] = []
    for i in range(num_consumers):
        thread = threading.Thread(
            target=consumer_worker,
            args=(file_queue, image_queue, stats),
            name=f"Consumer-{i+1}",
        )
        thread.start()
        consumers.append(thread)

    prod_thread.join()
    for consumer in consumers:
        consumer.join()

    # Сборка изображений из очереди
    raw_images: List[np.ndarray] = []
    while not image_queue.empty():
        raw_images.append(image_queue.get())

    # 3. Пул процессов (CPU-bound аугментация)
    augmented_results: List[np.ndarray] = []

    def result_callback(result: List[np.ndarray]) -> None:
        augmented_results.extend(result)

    with mp.Pool(processes=num_processes) as pool:
        for img in raw_images:
            pool.apply_async(augment_single_image, args=(img,), callback=result_callback)
        pool.close()
        pool.join()

    return stats.get_count(), len(augmented_results)