"""Модуль управления датасетом: загрузка, нормализация и разбиение."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from part1_mlp.app.exceptions import DataFormatError


class DatasetManager:
    """Управление табличными датасетами для обучения нейронной сети."""

    def __init__(self) -> None:
        self.features: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None

    def load_from_csv(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Загрузка числового датасета из CSV-файла."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"CSV файл не найден: {path}")

        try:
            raw_data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)

            if raw_data.shape[1] < 2:
                raise DataFormatError("Датасет должен содержать минимум 1 признак и 1 целевой столбец.")

            self.features = raw_data[:, :-1]
            self.targets = raw_data[:, -1:]
            return self.features, self.targets
        except Exception as error:
            raise DataFormatError(f"Не удалось корректно прочитать CSV: {error}") from error

    def normalize_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        """Нормализация Min-Max в диапазон [0, 1]."""
        if self.features is None or self.targets is None:
            raise ValueError("Данные еще не были загружены.")

        self.min_vals = np.min(self.features, axis=0)
        self.max_vals = np.max(self.features, axis=0)
        denom = np.where(self.max_vals - self.min_vals == 0.0, 1.0, self.max_vals - self.min_vals)

        self.features = (self.features - self.min_vals) / denom
        return self.features, self.targets

    def split_train_test(
        self, test_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разбиение данных на обучающую и тестовую выборки."""
        if self.features is None or self.targets is None:
            raise ValueError("Данные не загружены.")

        total_samples = self.features.shape[0]
        test_size = int(total_samples * test_ratio)

        shuffled_indices = np.random.permutation(total_samples)
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]

        x_train, y_train = self.features[train_indices], self.targets[train_indices]
        x_test, y_test = self.features[test_indices], self.targets[test_indices]

        return x_train, y_train, x_test, y_test