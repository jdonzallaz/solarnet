import logging

from colorlog import ColoredFormatter


def init_log():
    # Set logger
    LOG_LEVEL = logging.WARNING
    LOG_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)


def set_log_level(level: int):
    logger = logging.getLogger()
    logger.setLevel(level)
    lightning_logger = logging.getLogger("pytorch_lightning")
    lightning_logger.setLevel(level)
