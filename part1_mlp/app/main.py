"""Консольное меню для управления жизненным циклом нейросети."""

import sys
from pathlib import Path
from typing import Optional
import numpy as np

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from part1_mlp.app.exceptions import NeuralNetworkError  # noqa: E402
from part1_mlp.app.manager import DatasetManager  # noqa: E402
from part1_mlp.app.models import NeuralNetwork  # noqa: E402
from part1_mlp.app.utils import plot_loss  # noqa: E402


def print_menu() -> None:
    """Вывод пунктов меню в консоль."""
    print("\n" + "=" * 50)
    print("      ИНТЕЛЛЕКТУАЛЬНЫЙ ТЕРМИНАЛ: ЧАСТЬ 1 (MLP)")
    print("=" * 50)
    print("1. Создать сеть (размеры слоев, функция активации)")
    print("2. Загрузить данные из CSV")
    print("3. Обучить сеть (learning rate, эпохи, батч)")
    print("4. Сделать предсказание (ручной ввод)")
    print("5. Сохранить веса в JSON")
    print("6. Загрузить веса из JSON")
    print("7. Показать график функции потерь")
    print("0. Выход")
    print("=" * 50)


def main() -> None:
    network: Optional[NeuralNetwork] = None
    manager = DatasetManager()
    x_train, y_train, x_test, y_test = None, None, None, None

    while True:
        print_menu()
        choice = input("Выберите пункт меню (0-7): ").strip()

        try:
            if choice == "1":
                raw_layers = input("Введите размеры слоев через пробел (например, 2 4 1): ").strip()
                layers = [int(val) for val in raw_layers.split()]
                act = input("Выберите активацию скрытых слоев (sigmoid / relu) [sigmoid]: ").strip() or "sigmoid"
                network = NeuralNetwork(layer_sizes=layers, activation=act)
                print(f"[Успех] Нейронная сеть {layers} с активацией '{act}' создана!")

            elif choice == "2":
                csv_path = input("Введите путь к CSV файлу [data/input/sample_data.csv]: ").strip()
                if not csv_path:
                    csv_path = "data/input/sample_data.csv"
                manager.load_from_csv(csv_path)
                manager.normalize_min_max()
                x_train, y_train, x_test, y_test = manager.split_train_test(test_ratio=0.2)
                print(f"[Успех] Данные загружены и нормализованы! Train: {x_train.shape[0]} строк, Test: {x_test.shape[0]} строк.")

            elif choice == "3":
                if network is None:
                    print("[Ошибка] Сначала создайте сеть (пункт 1).")
                    continue
                if x_train is None or y_train is None:
                    print("[Ошибка] Сначала загрузите данные (пункт 2).")
                    continue

                lr = float(input("Скорость обучения (learning rate) [0.1]: ").strip() or "0.1")
                epochs = int(input("Число эпох [100]: ").strip() or "100")
                batch_size = int(input("Размер батча [4]: ").strip() or "4")

                print("\n--- Запуск процесса обучения ---")
                network.train(x_train, y_train, epochs=epochs, learning_rate=lr, batch_size=batch_size)
                print("--- Обучение успешно завершено! ---")

                if x_test is not None and len(x_test) > 0 and y_test is not None:
                    test_preds = network.predict(x_test)
                    test_mse = float(np.mean((test_preds - y_test) ** 2))
                    print(f"Ошибка MSE на тестовой выборке: {test_mse:.6f}")

            elif choice == "4":
                if network is None:
                    print("[Ошибка] Сеть еще не инициализирована.")
                    continue
                in_dim = network.layer_sizes[0]
                user_val = input(f"Введите {in_dim} признаков через пробел: ").strip()
                sample_features = np.array([float(v) for v in user_val.split()], dtype=np.float64)

                if manager.min_vals is not None and manager.max_vals is not None:
                    denom = np.where(manager.max_vals - manager.min_vals == 0.0, 1.0, manager.max_vals - manager.min_vals)
                    sample_features = (sample_features - manager.min_vals) / denom

                prediction = network.predict(sample_features)
                print(f"[Результат предсказания]: {prediction.flatten()}")

            elif choice == "5":
                if network is None:
                    print("[Ошибка] Сеть не создана.")
                    continue
                save_file = input("Имя файла для сохранения [data/output/weights.json]: ").strip() or "data/output/weights.json"
                network.save_weights(save_file)
                print(f"[Успех] Параметры сохранены в {save_file}")

            elif choice == "6":
                load_file = input("Имя файла для загрузки [data/output/weights.json]: ").strip() or "data/output/weights.json"
                if network is None:
                    network = NeuralNetwork(layer_sizes=[2, 2], activation="sigmoid")
                network.load_weights(load_file)
                print(f"[Успех] Веса успешно восстановлены! Архитектура: {network.layer_sizes}")

            elif choice == "7":
                if network is None or not network.loss_history:
                    print("[Предупреждение] История ошибок пуста. Сначала обучите сеть.")
                    continue
                plot_loss(network.loss_history)

            elif choice == "0":
                print("Завершение сеанса работы. До свидания!")
                break
            else:
                print("[Ошибка] Неверный пункт меню. Выберите от 0 до 7.")

        except (ValueError, FileNotFoundError, NeuralNetworkError) as err:
            print(f"[ОШИБКА ПРИЛОЖЕНИЯ]: {err}")


if __name__ == "__main__":
    main()