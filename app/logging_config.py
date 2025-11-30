import logging
from logging.handlers import RotatingFileHandler
from flask import Flask

from .config import Config


def setup_logging(app: Flask) -> None:
    """
    Attach a rotating file handler to app.logger,
    using paths from Config.
    """
    logger = app.logger
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    # Avoid duplicate handlers if reloaded
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(handler)
