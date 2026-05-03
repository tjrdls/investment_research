# -*- coding: utf-8 -*-
import logging
import sys


def configure_root_logger(level: int = logging.INFO) -> None:
    """애플리케이션 진입점(app.py, main.py)에서 한 번만 호출."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
