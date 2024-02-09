"""Logging module for ES based RAG."""

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

# Log format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(module)s: %(message)s"
LOG_FORMATTER = logging.Formatter(LOG_FORMAT)

log_path=os.path.join(os.getenv("LOG_DIR","log"))
isExist = os.path.exists(log_path)
if not isExist:
    # Create a new directory because it does not exist
   os.makedirs(log_path)
   print("The new Log is created!")

# Log file
LOG_FILE = os.path.join(os.getenv("LOG_DIR","log"), "rag.log")

LOG_FILE_HANDLER = RotatingFileHandler(
    LOG_FILE, maxBytes=1024 * 1024 * 10, backupCount=5, encoding='utf-8'
)
LOG_FILE_HANDLER.setFormatter(LOG_FORMATTER)
LOG_FILE_HANDLER.setLevel(logging.DEBUG)
LOG.addHandler(LOG_FILE_HANDLER)

# Log to stdout
'''
LOG_STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
LOG_STDOUT_HANDLER.setFormatter(LOG_FORMATTER)
LOG_STDOUT_HANDLER.setLevel(logging.INFO)
LOG.addHandler(LOG_STDOUT_HANDLER)
'''

def handle_error(e):
    LOG.error("%s: %s", str(e), type(e).__name__)
    traceback.print_exc()