"""
Logging configuration for the Error Warning and Trapping System (EWTS).

This module defines the centralized logging configuration used by EWTS.
It is responsible for creating and configuring a single, named logger
within the Python logging framework, based on environment variables
provided by the runtime environment (e.g., ngen).

Logging configuration is performed via configure_logging(), which applies
handlers, formatters, and log levels to the EWTS logger. The configuration
function is idempotent: once the logger has been initialized, subsequent
calls return immediately without modifying the existing configuration.

Configuration behavior is controlled by environment variables, whose names
are defined in constants.py:

    - EV_EWTS_LOGGING:
        Enables or disables EWTS logging. If set to "DISABLED", logging is
        disabled entirely for the EWTS logger. If unset, logging is enabled
        by default.

    - EV_MODULE_LOGLEVEL:
        Specifies the log level for the EWTS logger. Supported values include
        standard Python logging levels as well as ngen-style levels (e.g.,
        "SEVERE", "FATAL"), which are translated to Python equivalents.

Log output is directed to a file determined by the path-resolution utilities
in paths.py. If a log file cannot be created, logging falls back to stdout.

This module does not expose logging APIs directly; callers are expected to
retrieve the configured logger by name using logging.getLogger(MODULE_NAME).
"""

import logging
import sys
import os

from .constants import (
    MODULE_NAME,
    EV_EWTS_LOGGING,
    EV_MODULE_LOGLEVEL,
    LOG_MODULE_NAME_LEN,
)
from .formatter import CustomFormatter
from .paths import get_log_file_path
from .env import getenv_any

def translate_ngwpc_log_level(level: str) -> str:
    level = level.strip().upper()
    return {
        "SEVERE": "ERROR",
        "FATAL": "CRITICAL",
    }.get(level, level)


def force_info(handler, logger, msg, *args):
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        msg,
        args,
        None,
    )
    handler.emit(record)


def configure_logging():
    '''
    Set logging level and specify logger configuration based on environment variables set by ngen
    '''
    logger = logging.getLogger(MODULE_NAME)

    if getattr(logger, "_initialized", False):
        return logger # logger already initialized, nothing else to do

    # Default to enabled if flag not set or is set to disabled
    raw_value = getenv_any(EV_EWTS_LOGGING, "")
    normalized = (raw_value or "").strip().lower()  # convert None or "" to "", lowercase for easy comparison

    # Determine if logging is enabled
    enabled = normalized != "disabled"

    # Inform user if logging is enabled by default (env not explicitly set to "enabled")
    if enabled and normalized not in ("enabled",):
        print(f"{EV_EWTS_LOGGING} not explicitly set to 'ENABLED'; logging ENABLED by default", flush=True)

    if not enabled:
        logger.disabled = True
        logger._initialized = True
        print(f"Module {MODULE_NAME} Logging DISABLED", flush=True)
        return logger

    print(f"Module {MODULE_NAME} Logging ENABLED", flush=True)

    logFilePath, appendEntries = get_log_file_path()

    handler = (
        logging.FileHandler(logFilePath, mode="a" if appendEntries else "w")
        if logFilePath
        else logging.StreamHandler(sys.stdout)
    )

    log_level = translate_ngwpc_log_level(
        getenv_any(EV_MODULE_LOGLEVEL, "INFO")
    )

    module_fmt = MODULE_NAME.upper().ljust(LOG_MODULE_NAME_LEN)[:LOG_MODULE_NAME_LEN]

    formatter = CustomFormatter(
        fmt=f"%(asctime)s.%(msecs)03d {module_fmt} %(levelname_padded)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Setup logger
    logger.handlers.clear() # Clear any default handlers
    logger.setLevel(log_level)
    logger.addHandler(handler)

    # Write log level INFO message to log regradless of the actual log level
    force_info(handler, logger, "Log level set to %s", log_level)
    print(f"Module {MODULE_NAME} Log Level set to {log_level}", flush=True)

    logger._initialized = True
    return logger
