import logging
import subprocess
import sys
import os

LOG_PATH = '/var/www/html'
LOG_FILE_NAME = 'runner.log'
PRODUCTION_LOG_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] ' "\t" '%(message)s')

log_file_path = LOG_FILE_NAME
if os.path.exists(LOG_PATH):
    log_file_path = PRODUCTION_LOG_PATH

file_log = logging.FileHandler(log_file_path)
file_log.setFormatter(formatter)

stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_log)
logger.addHandler(stdout)
