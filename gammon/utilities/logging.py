# utilities/logging.py
import logging


def get_logger():
    logger = logging.getLogger("GCMC")
    if not logger.hasHandlers():
        logger.setLevel(logging.WARNING)

        # Stream handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    return logger


def set_log_debug_mode():
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    # TODO Fails if the user defined his handler
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    file_handler = logging.FileHandler('Debug.log')
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    file_formatter = logging.Formatter(fmt)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
