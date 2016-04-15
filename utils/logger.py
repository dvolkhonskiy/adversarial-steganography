import logging
import subprocess
import sys
import os

LOG_FILE_NAME = 'main.log'

formatter = logging.Formatter('%(asctime)s [%(levelname)s] ' "\t" '%(message)s')

log_file_path = LOG_FILE_NAME

file_log = logging.FileHandler(log_file_path)
file_log.setFormatter(formatter)

stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_log)
logger.addHandler(stdout)


def log(message):
    # decorator for logging what function is doing
    def wrapper(func):
        def execute(*args, **kwargs):
            logger.debug(message)
            result = func(*args, **kwargs)
            return result
        return execute
    return wrapper