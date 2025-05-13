import os

from .logger import get_logger


def setup_env():
    # Get logger
    logger = get_logger()

    # Disable rust logs by default
    if os.environ.get("RUST_LOG") is None:
        logger.info("Disabled rust logs")
        os.environ["RUST_LOG"] = "off"