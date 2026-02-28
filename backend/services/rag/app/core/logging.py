from __future__ import annotations

import logging
import sys


# -----------------------------------------------------------------------------
# Structured logging — outputs key=value pairs to stdout.
# Used consistently across all pipeline modules.
# -----------------------------------------------------------------------------
class StructuredFormatter(logging.Formatter):
    """Formats log records as structured key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra context passed via the `extra` kwarg.
        _SKIP = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "id", "levelname", "levelno",
            "lineno", "message", "module", "msecs", "msg", "name",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _SKIP:
                fields[key] = value

        return " | ".join(f"{k}={v}" for k, v in fields.items())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the StructuredFormatter attached."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger
