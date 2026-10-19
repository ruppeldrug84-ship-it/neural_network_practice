"""Модуль трансформаций и аугментации данных (CPU-bound операции)."""

from typing import List
import numpy as np


def augment_single_image(image_array: np.ndarray) -> List[np.ndarray]:
    """Применение набора аугментаций к массиву изображения.

    :param image_array: Входное изображение в формате NumPy массива.
    :return: Список модифицированных изображений (оригинал, поворот, флип, шум).
    """
    augmented_list: List[np.ndarray] = [image_array]

    # Поворот на 90 градусов
    rotated = np.rot90(image_array, k=1, axes=(0, 1))
    augmented_list.append(rotated)

    # Горизонтальное отражение
    flipped = np.fliplr(image_array)
    augmented_list.append(flipped)

    # Наложение гауссова шума
    noise = np.random.normal(loc=0.0, scale=0.04, size=image_array.shape).astype(np.float32)
    noisy_image = np.clip(image_array + noise, 0.0, 1.0)
    augmented_list.append(noisy_image)

    return augmented_list