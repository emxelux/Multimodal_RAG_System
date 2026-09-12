import os
import logging.config

# Create target logs directory if it does not exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
            "level": "INFO",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(LOGS_DIR, "error.log"),  # Absolute path to logs/
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "default",
            "level": "ERROR",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)