import multiprocessing

# Gunicorn config variables
workers_per_core_str = "1"
max_workers_str = "1"
bind = "0.0.0.0:8000"
graceful_timeout = 120
timeout = 120
keepalive = 5

# For debugging and testing
log_level = "info"
worker_tmp_dir = "/dev/shm"

# Calculate number of workers based on cores
workers = multiprocessing.cpu_count()
if workers > int(max_workers_str):
    workers = int(max_workers_str)

# Gunicorn log configuration
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "generic": {
            "format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
            "datefmt": "[%Y-%m-%d %H:%M:%S %z]",
            "class": "logging.Formatter",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "gunicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "gunicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
} 