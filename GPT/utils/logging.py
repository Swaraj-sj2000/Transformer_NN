#GPT/utils/logging.py
import logging
import os

os.makedirs("GPT/logs",exist_ok=True)

def get_logger(name):

    logger=logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter=logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    file_handler=logging.FileHandler(
        f"logs/{name}.log"
    )

    file_handler.setFormatter(formatter)

    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger