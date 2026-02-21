import logging
import sys

from loguru import logger
from selenium.webdriver.remote.remote_connection import LOGGER as selenium_logger

from config import LOG_LEVEL, LOG_SELENIUM_LEVEL, LOG_TO_CONSOLE


def remove_default_loggers():
    """Remove default handlers from Python root logger."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()


def init_loguru_logger():
    """Initialize loguru with console logging only."""
    logger.remove()

    if LOG_TO_CONSOLE:
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            backtrace=True,
            diagnose=True,
        )


def init_selenium_logger():
    """Configure selenium logger level without file sink."""
    selenium_logger.handlers.clear()
    selenium_logger.setLevel(LOG_SELENIUM_LEVEL)


remove_default_loggers()
init_loguru_logger()
init_selenium_logger()
