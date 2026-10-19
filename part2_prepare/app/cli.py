"""Точка входа CLI утилиты препроцессинга и диагностики окружения."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

# Обеспечиваем корректный импорт модулей проекта
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from part2_prepare.app.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_PATH,
)
from part2_prepare.app.logging_config import setup_logger  # noqa: E402
from part2_prepare.app.scanner import process_images_and_tables  # noqa: E402

logger = setup_logger("cli")


def run_doctor() -> None:
    """Проверка рабочего окружения: версии библиотек, Python, рабочей папки и CUDA."""
    print("\n" + "=" * 55)
    print("      ДИАГНОСТИКА ОКРУЖЕНИЯ СИСТЕМЫ (DOCTOR)")
    print("=" * 55)

    print(f"[OK] Версия Python: {sys.version.split()[0]}")
    print(f"[OK] Текущая рабочая директория: {Path.cwd()}")

    libraries = ["numpy", "Pillow", "matplotlib", "python-dotenv"]
    for lib in libraries:
        try:
            ver = importlib.metadata.version(lib)
            print(f"[OK] Библиотека {lib}: версия {ver}")
        except importlib.metadata.PackageNotFoundError:
            print(f"[!] Библиотека {lib} не установлена")

    # Проверка доступности CUDA через subprocess.run
    print("-" * 55)
    print("Проверка аппаратного ускорения NVIDIA CUDA:")
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            gpu_line = [line for line in proc.stdout.split("\n") if "NVIDIA" in line or "GeForce" in line]
            gpu_name = gpu_line[0].strip() if gpu_line else "NVIDIA GPU обнаружен"
            print(f"[OK] CUDA / GPU активна: {gpu_name}")
        else:
            print("[INFO] Утилита nvidia-smi вернула код ошибки. Работа в режиме CPU.")
    except FileNotFoundError:
        print("[INFO] nvidia-smi не обнаружен в PATH (работа выполняется на процессоре CPU).")
    print("=" * 55 + "\n")


def main() -> None:
    """Парсинг аргументов командной строки и маршрутизация команд."""
    parser = argparse.ArgumentParser(
        prog="preprocessor",
        description="CLI-утилита для препроцессинга датасетов и диагностики окружения.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Подкоманда prepare
    prep_parser = subparsers.add_parser("prepare", help="Подготовка и нормализация набора данных в .npy")
    prep_parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=f"Путь к директории с файлами (по умолчанию: {DEFAULT_DATA_PATH})",
    )
    prep_parser.add_argument(
        "--ext",
        nargs="+",
        default=["csv"],
        help="Список расширений для обработки через пробел (например: --ext csv или --ext png jpg)",
    )
    prep_parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Путь к итоговому .npy файлу (по умолчанию: {DEFAULT_OUTPUT_PATH})",
    )

    # Подкоманда doctor
    subparsers.add_parser("doctor", help="Проверка версий Python, библиотек и доступности CUDA")

    args = parser.parse_args()

    if args.command == "doctor":
        run_doctor()
    elif args.command == "prepare":
        source_dir = Path(args.path)
        output_file = Path(args.output)
        try:
            count, shape, duration = process_images_and_tables(
                source_dir=source_dir,
                extensions=args.ext,
                output_file=output_file,
            )
            print("\n" + "=" * 50)
            print("         ИТОГОВАЯ СТАТИСТИКА ОБРАБОТКИ")
            print("=" * 50)
            print(f"Обработано файлов:   {count}")
            print(f"Размерность массива: {shape}")
            print(f"Время выполнения:    {duration:.4f} сек.")
            print(f"Результат сохранен:  {output_file}")
            print("=" * 50 + "\n")
        except Exception as exc:
            logger.error("Критическая ошибка при выполнении prepare: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()