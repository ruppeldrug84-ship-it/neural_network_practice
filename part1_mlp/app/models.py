"""Модуль объектно-ориентированной реализации многослойного перцептрона (MLP)."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np

from part1_mlp.app.exceptions import (
    DataFormatError,
    InvalidLayerSizeError,
    MismatchedDataError,
    ModelNotTrainedError,
)


class NeuralNetwork:
    """Многослойный перцептрон с поддержкой градиентного спуска и батчей."""

    def __init__(self, layer_sizes: List[int], activation: str = "sigmoid") -> None:
        """Инициализация архитектуры нейронной сети.

        :param layer_sizes: Список с размерами слоев (входной, скрытые, выходной).
        :param activation: Функция активации скрытых слоев ('sigmoid' или 'relu').
        """
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Сеть должна содержать минимум входной и выходной слои.")

        for size in layer_sizes:
            if not isinstance(size, int) or size <= 0:
                raise InvalidLayerSizeError(f"Размер слоя должен быть > 0, получено: {size}")

        self.layer_sizes: List[int] = layer_sizes
        self.activation_name: str = activation.lower()
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.loss_history: List[float] = []

        self._init_weights()

    def _init_weights(self) -> None:
        """Инициализация весов He / Xavier."""
        self.weights.clear()
        self.biases.clear()

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / in_dim)
            else:
                scale = np.sqrt(1.0 / in_dim)

            weight_matrix = np.random.randn(in_dim, out_dim) * scale
            bias_vector = np.zeros((1, out_dim), dtype=np.float64)

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def _activate(self, z: np.ndarray, is_output: bool = False) -> np.ndarray:
        """Применение функции активации."""
        if is_output or self.activation_name == "sigmoid":
            clipped_z = np.clip(z, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-clipped_z))
        elif self.activation_name == "relu":
            return np.maximum(0.0, z)
        raise ValueError(f"Неизвестная функция активации: {self.activation_name}")

    def _activate_derivative(self, a: np.ndarray, is_output: bool = False) -> np.ndarray:
        """Производная функции активации."""
        if is_output or self.activation_name == "sigmoid":
            return a * (1.0 - a)
        elif self.activation_name == "relu":
            return (a > 0.0).astype(np.float64)
        raise ValueError(f"Неизвестная функция активации: {self.activation_name}")

    def forward(self, inputs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Прямой проход (forward pass)."""
        activations: List[np.ndarray] = [inputs]
        linear_outputs: List[np.ndarray] = []

        current_a = inputs
        total_layers = len(self.weights)

        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_a, w) + b
            linear_outputs.append(z)
            is_last = layer_idx == total_layers - 1
            current_a = self._activate(z, is_output=is_last)
            activations.append(current_a)

        return linear_outputs, activations

    def backward(
        self,
        activations: List[np.ndarray],
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        """Обратный проход (backpropagation) и градиентный спуск."""
        batch_size = targets.shape[0]
        output_activation = activations[-1]

        loss = float(np.mean((output_activation - targets) ** 2))
        delta = (output_activation - targets) * self._activate_derivative(output_activation, is_output=True)

        total_layers = len(self.weights)
        for layer_idx in reversed(range(total_layers)):
            prev_a = activations[layer_idx]
            grad_w = np.dot(prev_a.T, delta) / batch_size
            grad_b = np.sum(delta, axis=0, keepdims=True) / batch_size

            if layer_idx > 0:
                next_delta = np.dot(delta, self.weights[layer_idx].T)
                delta = next_delta * self._activate_derivative(prev_a, is_output=False)

            self.weights[layer_idx] -= learning_rate * grad_w
            self.biases[layer_idx] -= learning_rate * grad_b

        return loss

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int,
    ) -> List[float]:
        """Обучение на батчах с фиксацией MSE."""
        if features.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Размерность данных ({features.shape[1]}) не совпадает со входом сети ({self.layer_sizes[0]})."
            )

        num_samples = features.shape[0]
        self.loss_history.clear()

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            shuffled_x = features[indices]
            shuffled_y = targets[indices]

            epoch_losses: List[float] = []

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_x = shuffled_x[start_idx:end_idx]
                batch_y = shuffled_y[start_idx:end_idx]

                _, activations = self.forward(batch_x)
                batch_loss = self.backward(activations, batch_y, learning_rate)
                epoch_losses.append(batch_loss)

            mean_epoch_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_epoch_loss)

            if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"Эпоха {epoch:4d}/{epochs:4d} | Ошибка MSE: {mean_epoch_loss:.6f}")

        return self.loss_history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Предсказание для новых примеров."""
        if not self.weights:
            raise ModelNotTrainedError("Модель не инициализирована.")
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидалось признаков: {self.layer_sizes[0]}, получено: {inputs.shape[1]}"
            )

        _, activations = self.forward(inputs)
        return activations[-1]

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """Сохранение весов в JSON."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "layer_sizes": self.layer_sizes,
            "activation_name": self.activation_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """Загрузка весов из JSON."""
        load_path = Path(filepath)
        if not load_path.is_file():
            raise FileNotFoundError(f"Файл параметров не найден: {load_path}")

        try:
            with open(load_path, "r", encoding="utf-8") as file:
                payload: Dict = json.load(file)

            self.layer_sizes = payload["layer_sizes"]
            self.activation_name = payload["activation_name"]
            self.weights = [np.array(w, dtype=np.float64) for w in payload["weights"]]
            self.biases = [np.array(b, dtype=np.float64) for b in payload["biases"]]
        except Exception as error:
            raise DataFormatError(f"Ошибка при разборе файла параметров: {error}") from error