"""
Error Warning and Trapping System (EWTS) Package API

This package provides a centralized, named logging configuration for the
Error, Warning, and Trapping System used throughout the codebase.

EWTS configures a single, shared logger in the Python logging framework,
identified by a fixed module name. All modules that participate in EWTS
logging retrieve this logger by name via the standard logging API.

Logging configuration should be performed once at application startup by
calling configure_logging(). The configuration function is idempotent:
subsequent calls have no effect and will not reconfigure handlers or levels.

The logger name is exposed to allow any module to obtain the configured
logger without importing internal implementation details.

Typical usage:

    At application startup:
        from troute_ewts import configure_logging
        configure_logging()

    Within other modules:
        import logging
        from troute_ewts import MODULE_NAME

        LOG = logging.getLogger(MODULE_NAME)
"""

from .constants import MODULE_NAME
from .config import configure_logging

__all__ = ["MODULE_NAME", "configure_logging"]
