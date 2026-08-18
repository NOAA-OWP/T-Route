"""
Constants and configuration keys for the Error Warning and Trapping System (EWTS).

This module defines all constant values used by EWTS for logging configuration,
environment variable integration, and log file naming. These values represent
the stable interface between EWTS, ngen, and participating Python modules.

Constants are grouped into two categories:

    1) Module-specific constants:
       Values that uniquely identify the current ngen module, including the
       logger name and module-specific environment variables.

    2) Common constants:
       Values shared across ngen modules that control global logging behavior,
       filesystem layout, and integration with the ngen runtime environment.

These constants are intentionally centralized to ensure consistent behavior
across the codebase and to avoid hard-coded strings in implementation logic.
Callers should treat these values as read-only.
"""


# Values unique to each ngen module
MODULE_NAME           = "T-Route"
EV_MODULE_LOGLEVEL    = "TROUTE_LOGLEVEL"      # This modules log level
EV_MODULE_LOGFILEPATH = "TROUTE_LOGFILEPATH"   # This modules log full log filename

# Values common to all ngen modules
EV_NGEN_LOGFILEPATH   = "NGEN_LOG_FILE_PATH"   # Environment variable name with the log file location typically set by ngen
EV_EWTS_LOGGING       = "NGEN_EWTS_LOGGING"    # Environment variable name with the enable/disable state for the Error Warning  
                                               # and Trapping System typically set by ngen

DS                    = "/"                    # Directory separator
LOG_DIR_DEFAULT       = "run-logs"             # Default parent log directory string if env var empty & ngencerf doesn't exist
LOG_DIR_NGENCERF      = "/ngencerf/data"       # ngenCERF log directory string if environement var empty.
LOG_FILE_EXT          = "log"                  # Log file name extension
LOG_MODULE_NAME_LEN   = 8                      # Width of module name for log entries


