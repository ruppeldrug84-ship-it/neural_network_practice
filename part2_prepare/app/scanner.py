"""Модуль рекурсивного сканирования директорий и преобразования данных в NumPy."""

import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

from part2_prepare.app.logging_config import setup_logger

logger = setup_logger("scanner")


def process_images_and_tables(
    source_dir: Path,
    extensions: List[str],
    output_file: Path,
    target_img_size: Tuple[int, int] = (64, 64),
) -> Tuple[int, Tuple[int, ...], float]:
    """Рекурсивный поиск файлов, их нормализация и сохранение в формат .npy.

    :param source_dir: Директория для поиска файлов.
    :param extensions: Список расширений для поиска (например, ['.png', '.jpg', '.csv']).
    :param output_file: Путь к итоговому файлу .npy.
    :param target_img_size: Размер, к которому приводятся изображения.
    :return: Кортеж (количество файлов, размерность итогового массива, время обработки в секундах).
    """
    start_time = time.perf_counter()

    if not source_dir.exists():
        raise FileNotFoundError(f"Исходная директория не существует: {source_dir}")

    normalized_extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]
    logger.info("Сканирование каталога %s для расширений: %s", source_dir, normalized_extensions)

    found_files: List[Path] = []
    for file_path in source_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in normalized_extensions:
            found_files.append(file_path)

    if not found_files:
        logger.warning("Файлы с указанными расширениями не найдены.")
        return 0, (), 0.0

    logger.info("Найдено файлов: %d. Начинается обработка...", len(found_files))

    data_list: List[np.ndarray] = []
    for file_path in found_files:
        ext = file_path.suffix.lower()
        try:
            if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                with Image.open(file_path) as img:
                    img_rgb = img.convert("RGB")
                    img_resized = img_rgb.resize(target_img_size)
                    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
                    data_list.append(arr)
            elif ext == ".csv":
                raw_table = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=np.float32)
                if raw_table.ndim == 1:
                    raw_table = raw_table.reshape(1, -1)
                # Нормализация таблицы
                col_min = np.min(raw_table, axis=0)
                col_max = np.max(raw_table, axis=0)
                denom = np.where(col_max - col_min == 0.0, 1.0, col_max - col_min)
                norm_table = (raw_table - col_min) / denom
                data_list.append(norm_table)
        except Exception as error:
            logger.error("Ошибка при обработке файла %s: %s", file_path, error)

    if not data_list:
        raise ValueError("Не удалось преобразовать ни один файл.")

    # Если обрабатывались одинаковые изображения — объединяем в один массив
    try:
        combined_array = np.array(data_list)
    except ValueError:
        # Для разнородных таблиц объединяем по строкам
        combined_array = np.vstack(data_list)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, combined_array)

    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "Обработка завершена! Сохранено в %s | Размер: %s | Время: %.4f сек.",
        output_file,
        combined_array.shape,
        elapsed_time,
    )

    return len(found_files), combined_array.shape, elapsed_time