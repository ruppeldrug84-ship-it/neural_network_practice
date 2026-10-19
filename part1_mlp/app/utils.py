"""Вспомогательные утилиты для визуализации результатов."""

from typing import List
import matplotlib.pyplot as plt


def plot_loss(loss_history: List[float], title: str = "Динамика ошибки MSE по эпохам") -> None:
    """Отображение графика функции потерь через Matplotlib."""
    if not loss_history:
        print("История ошибок пуста, график не может быть построен.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss (MSE)", color="#1f77b4", lw=2)
    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()