import sys

from loguru import logger as loguru_logger

_LOGGER = None


def setup_logger(rank: int, log_level: str = "INFO"):
    """Configure multi-process logger shared across modules"""
    global _LOGGER
    assert _LOGGER is None, "Logger already setup"
    loguru_logger.remove()  # Remove default handlers
    loguru_logger.add(
        sys.stdout,
        format=f"[Rank {rank}] <green>{{time:YYYY-MM-DD HH:mm:ss}}</green> | <level>{{level}}</level> | <cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:<cyan>{{line}}</cyan> - <level>{{message}}</level>",
        colorize=True,
        enqueue=True,  # Ensures thread/process safety
        level=log_level,
    )

    _LOGGER = loguru_logger.bind(rank=rank)
    return _LOGGER


def get_logger():
    """Get global logger instance."""
    global _LOGGER
    assert _LOGGER is not None, "Logger not setup"
    return _LOGGER
