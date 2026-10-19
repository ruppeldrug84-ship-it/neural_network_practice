"""Модуль управления конфигурацией приложения через переменные окружения."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env, если файл существует
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path)

# Значения конфигурации с безопасными значениями по умолчанию
DEFAULT_DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "input")))
DEFAULT_OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(BASE_DIR / "data" / "output" / "preprocessed.npy")))
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()