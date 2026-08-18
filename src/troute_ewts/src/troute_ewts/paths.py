"""
Log file path resolution utilities for the Error Warning and Trapping System (EWTS).

This module provides helper functions for constructing and validating log file
paths used by the EWTS logging configuration. Log file selection follows a
well-defined precedence based on environment variables and runtime availability.

Log file path precedence:

    1. If the NGEN-provided log file path is available via the environment variable
       defined in EV_NGEN_LOGFILEPATH, use that path.

    2. Otherwise, create a default, module-specific log file:
        2.1) Create a base log directory under the ngenCERF data directory if it
             exists; otherwise fall back to the user's home directory.
        2.2) Create a child directory using the current username if available,
             otherwise use the current UTC date (YYYYMMDD).
        2.3) Construct a log filename using the module name and a UTC timestamp.

The resolved log file path is validated by attempting to open the file. Upon
successful creation or reuse, the full log file path is stored in the
EV_MODULE_LOGFILEPATH environment variable so subsequent calls reuse the same
file. If log file creation fails, entries will be written to stdout.

This module does not configure loggers directly; it only resolves filesystem
paths and associated metadata required by the logging configuration layer.
"""

import getpass
import os
from datetime import datetime, timezone

from .env import getenv_any
from .constants import (
    MODULE_NAME,
    EV_NGEN_LOGFILEPATH,
    EV_MODULE_LOGFILEPATH,
    DS,
    LOG_DIR_DEFAULT,
    LOG_DIR_NGENCERF,
    LOG_FILE_EXT,
)

def create_timestamp(date_only=False, iso=False, append_ms=False):
    now = datetime.now(timezone.utc)

    if date_only:
        ts = now.strftime("%Y%m%d")
    elif iso:
        ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        ts = now.strftime("%Y%m%dT%H%M%S")

    if append_ms:
        ts += f".{now.microsecond // 1000:03d}"

    return ts

def get_log_file_path():
    # Determine the log file path using the following precedence:
    # 1) Use the ngen-provided log file path if available in the NGEN_LOG_FILE_PATH environment variable
    # 2) Otherwise, create a default module-specific log file using the module name and a UTC timestamp.
    # 2.1) First create a subdirectory under the ngenCERF data directory if available, otherwise the user home directory.
    # 2.2) Next create a subdirectory name using the username, if available, otherwise use the YYYYMMDD.
    # 2.3) Attempt to open the log file and upon failure, use stdout.

    appendEntries = True
    moduleLogFileExists = False

     # Determine if a log file has already been opened for this module (either the ngen log or default)
    moduleEnvVar = (os.getenv(EV_MODULE_LOGFILEPATH) or "").strip()
    if moduleEnvVar:
        logFilePath = moduleEnvVar
        moduleLogFileExists = True
    else:
        ngenEnvVar = getenv_any(EV_NGEN_LOGFILEPATH).strip()
        if ngenEnvVar:
            logFilePath = ngenEnvVar
        else:
            print(f"Module {MODULE_NAME} Env var {EV_NGEN_LOGFILEPATH} not found. Creating default log name.")
            appendEntries = False
            baseDir = (
                f"{LOG_DIR_NGENCERF}{DS}{LOG_DIR_DEFAULT}"
                if os.path.isdir(LOG_DIR_NGENCERF)
                else f"{os.path.expanduser('~')}{DS}{LOG_DIR_DEFAULT}"
            )
            try:
                os.makedirs(baseDir, exist_ok=True)

                childDir = getpass.getuser() or create_timestamp(True)
                logFileDir = f"{baseDir}{DS}{childDir}"
                os.makedirs(logFileDir, exist_ok=True)

                logFilePath = (
                    f"{logFileDir}{DS}{MODULE_NAME}_{create_timestamp()}.{LOG_FILE_EXT}"
                )
            except Exception as e:
                print(f"Module {MODULE_NAME} {e}", flush=True)
                logFilePath = ""
    
    # Ensure log file can be opened and set module env var
    try:
        if (logFilePath):
            mode = "a" if appendEntries else "w"
            with open(logFilePath, mode):
                pass
            if not moduleLogFileExists:
                os.environ[EV_MODULE_LOGFILEPATH] = logFilePath
                print(f"Module {MODULE_NAME} Log File: {logFilePath}", flush=True)
        else:
            raise IOError
    except Exception:
        print(f"Module {MODULE_NAME} Unable to open log file: {logFilePath}", flush=True)
        print(f"Module {MODULE_NAME} Log entries will be writen to stdout", flush=True)

    return logFilePath, appendEntries
