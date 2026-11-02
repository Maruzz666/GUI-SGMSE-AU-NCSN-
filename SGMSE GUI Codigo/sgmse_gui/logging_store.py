import logging
from logging.handlers import RotatingFileHandler

from .runtime import prepare_runtime


def create_logger() -> logging.Logger:
    logger = logging.getLogger("sgmse_gui")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            prepare_runtime() / "logs" / "sgmse-gui.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    return logger
