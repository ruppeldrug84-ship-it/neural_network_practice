"""Модуль централизованной настройки логирования."""

import logging
from pathlib import Path
from part2_prepare.app.config import BASE_DIR, DEFAULT_LOG_LEVEL


def setup_logger(name: str = "preprocessor") -> logging.Logger:
    """Инициализация и конфигурирование логгера.

    :param name: Имя логгера.
    :return: Сконфигурированный объект logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger