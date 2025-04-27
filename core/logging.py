# core/logging.py
# Source: Based on legacy codebase.txt
"""
Configures and provides logging functionality for the application.
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up logging based on the provided configuration dictionary using dictConfig.

    Args:
        config (Dict[str, Any]): The configuration dictionary, expected to
                                 contain 'log_level' and 'output_dir'.
    """
    log_level_str: str = config.get("log_level", "INFO").upper()
    log_dir_str: str = config.get("output_dir", "output")
    log_filename: str = "app.log"

    log_dir_path = Path(log_dir_str)
    log_file_path = log_dir_path / log_filename

    # Ensure the log directory exists
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Cannot log this error to file yet, print to console
        print(f"ERROR: Failed to create log directory {log_dir_path}: {e}")

    # Define logging configuration dictionary
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'file': {
                'level': log_level_str,
                'class': 'logging.FileHandler',
                'filename': str(log_file_path),
                'formatter': 'standard',
            },
            'console': {
                'level': log_level_str,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['file', 'console'],
                'level': log_level_str,
                'propagate': True,
            },
            'transformers': {
                'level': 'WARNING',  # Prevent libraries from flooding logs
            },
        },
    }

    try:
        # Apply the configuration
        logging.config.dictConfig(logging_config)
        logging.getLogger(__name__).info("Logging setup complete.")
    except Exception as e:
        # If dictConfig fails, print error to console
        print(f"ERROR: Failed to configure logging: {e}")


# These wrapper functions are maintained for compatibility with existing code
# but should be gradually replaced with direct logger usage

def log_info(message: str) -> None:
    """Logs an informational message."""
    caller_logger = logging.getLogger("core.logging")
    caller_logger.info(message)


def log_error(message: str) -> None:
    """Logs an error message."""
    caller_logger = logging.getLogger("core.logging")
    caller_logger.error(message)


def log_warning(message: str) -> None:
    """Logs a warning message."""
    caller_logger = logging.getLogger("core.logging")
    caller_logger.warning(message)
