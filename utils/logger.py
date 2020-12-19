import os
import logging


def set_logger():
    logger_format = "%(asctime)s [%(filename)s %(lineno)d %(levelname)s]: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=logger_format)
    logger = logging.getLogger(__name__)
    logger.info("start print log")


if __name__ == "__main__":
    set_logger()