"""
Custom log record formatting for the Error Warning and Trapping System (EWTS).

This module defines a custom logging formatter used by EWTS to produce
consistent, ngen-compatible log output across all participating modules.

The formatter applies the following behaviors:

    - Forces all timestamps to UTC, independent of system locale settings.
    - Formats timestamps with millisecond precision.
    - Maps Python logging levels to ngen-style severity names
      (e.g., ERROR → SEVERE, CRITICAL → FATAL).
    - Pads and normalizes level names to fixed width for column alignment.
    - Strips trailing whitespace and newline characters from log messages.

The formatter operates entirely within the Python logging framework and does
not modify logger configuration or handler behavior. It is intended to be used
by the EWTS logging configuration layer and not instantiated directly by
application code.
"""

import logging
import time

class CustomFormatter(logging.Formatter):
    LEVEL_NAME_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "SEVERE",
        logging.CRITICAL: "FATAL"
    }

    # Apply custom formatter (UTC timestamps applied only to this formatter)
    def converter(self, timestamp):
        """Override time converter to return UTC time tuple"""
        return time.gmtime(timestamp)

    def formatTime(self, record, datefmt=None):
        """Use our UTC converter"""
        ct = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, ct)
        t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return f"{t},{int(record.msecs):03d}"

    def format(self, record):
        # Strip trailing whitespace/newlines from the message
        if record.msg:
            record.msg = str(record.msg).rstrip()

        # Map level names
        original_levelname = record.levelname
        record.levelname = self.LEVEL_NAME_MAP.get(record.levelno, original_levelname)
        record.levelname_padded = record.levelname.ljust(7)[:7]  # Exactly 7 chars
        formatted = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname  # Restore original in case it's reused
        return formatted
