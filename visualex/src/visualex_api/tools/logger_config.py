import logging
import logging.config
import sys
import os
import structlog
from structlog.types import Processor

def setup_logging():
    # Correctly determine the project root to place the logs directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_dir = os.path.join(project_root, "src", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # These are the processors that will be used by structlog
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Configure standard logging
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": shared_processors,
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(),
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "console",
            },
            "file": {
                "level": "INFO",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": os.path.join(log_dir, "visualex.log"),
                "maxBytes": 1024 * 1024 * 5, # 5 MB
                "backupCount": 5,
                "formatter": "json",
            },
            "error_file": {
                "level": "ERROR",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": os.path.join(log_dir, "visualex_error.log"),
                "maxBytes": 1024 * 1024 * 5, # 5 MB
                "backupCount": 5,
                "formatter": "json",
            },
        },
        "loggers": {
            "": { # root logger
                "handlers": ["default", "file", "error_file"],
                "level": "INFO",
                "propagate": True,
            },
        }
    })

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
