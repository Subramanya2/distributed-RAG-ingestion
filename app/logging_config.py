"""
Structured JSON logging configuration.

All logs are emitted as single-line JSON objects, making them parseable
by log aggregators (ELK, Loki, CloudWatch, Datadog).
"""

import logging
import logging.config


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "app.logging_config.JsonFormatter",
        },
        "simple": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    "loggers": {
        "app": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
        "celery": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


class JsonFormatter(logging.Formatter):
    """
    Emit log records as single-line JSON objects.

    Output example::

        {"ts":"2024-01-15T10:30:00","level":"INFO","logger":"app.tasks","msg":"Document abc ingested."}
    """

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "task_id"):
            log_entry["task_id"] = record.task_id
        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """Apply the structured logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)
