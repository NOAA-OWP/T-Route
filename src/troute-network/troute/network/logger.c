#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <ctype.h>
#include <stdbool.h>

// Define Environment Variable Names
#define MODULE_NAME           "T-Route"
#define LOG_DIR_NGENCERF      "/ngencerf/data"        // ngenCERF log directory
#define LOG_DIR_DEFAULT       "run-logs"              // Default subdir if ngen log file env not found
#define LOG_FILE_EXT          "log"                   // Log file name extension
#define DS                    "/"                     // Directory separator
#define LOG_MODULE_NAME_LEN   8                       // Width of module name for log entries

#define EV_EWTS_LOGGING       "NGEN_EWTS_LOGGING"     // Enable/disable of Error Warning and Trapping System  
#define EV_NGEN_LOGFILEPATH   "NGEN_LOG_FILE_PATH"    // ngen log file 
#define EV_MODULE_LOGLEVEL    "TROUTE_LOGLEVEL"       // This modules log level
#define EV_MODULE_LOGFILEPATH "TROUTE_LOGFILEPATH"    // This modules log full log filename


bool     loggingEnabled = true;
bool     openedAppendMode = true;
bool     loggerInitialized = false;
FILE*    logFile = NULL;
char     logFilePath[1024] = "";
LogLevel logLevel = INFO;
char     moduleName[LOG_MODULE_NAME_LEN+1] = "";

void TrimString(const char *input, char *output, size_t outputSize) {
    if (!input || !output || outputSize == 0) {
        if (output && outputSize > 0) output[0] = '\0';
        return;
    }

    // Skip leading whitespace
    while (isspace((unsigned char)*input)) {
        input++;
    }

    size_t len = strlen(input);
    if (len == 0) {
        output[0] = '\0';
        return;
    }

    // Find the end of the string and move backward past trailing whitespace
    const char *end = input + len - 1;
    while (end > input && isspace((unsigned char)*end)) {
        end--;
    }

    size_t trimmedLen = end - input + 1;
    if (trimmedLen >= outputSize) {
        trimmedLen = outputSize - 1;
    }

    strncpy(output, input, trimmedLen);
    output[trimmedLen] = '\0';
}

void SetLogModuleName(void) {

    // Copy MODULE_NAME to a string variable 
    char src[20];
    strncpy(src, MODULE_NAME, sizeof(src));
    src[sizeof(src) - 1] = '\0'; // ensure null termination

    // Convert to uppercase and copy, up to width or null terminator
    size_t i = 0;
    for (; i < LOG_MODULE_NAME_LEN && src[i] != '\0'; i++) {
        moduleName[i] = toupper((unsigned char)src[i]);
    }

    // Pad with spaces if needed
    for (; i < LOG_MODULE_NAME_LEN; i++) {
        moduleName[i] = ' ';
    }
    moduleName[LOG_MODULE_NAME_LEN] = '\0'; // null-terminate
}

void CreateTimestamp(char *buffer, size_t size, bool appendMS, bool iso) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm utc_tm;
    gmtime_r(&tv.tv_sec, &utc_tm);

    char base[32];
    if (iso) {
        strftime(base, sizeof(base), "%Y-%m-%dT%H:%M:%S", &utc_tm);
    } else {
        strftime(base, sizeof(base), "%Y%m%dT%H%M%S", &utc_tm);
    }

    if (appendMS) {
        snprintf(buffer, size, "%s.%03ld", base, tv.tv_usec / 1000);
    } else {
        snprintf(buffer, size, "%s", base);
    }
}

bool DirectoryExists(const char *path) {
    struct stat info;
    return (stat(path, &info) == 0 && (info.st_mode & S_IFDIR));
}

bool CreateDirectory(const char *path) {
    if (!DirectoryExists(path)) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", path);
        int status = system(cmd);
        if (status == -1 || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
            fprintf(stderr, "Failed to create directory: %s\n", path);
            return false;
        }
    }
    return true;
}

bool LogFileReady(bool appendMode) {
    if (logFile && !ferror(logFile)) {
        fseek(logFile, 0, SEEK_END);
        return true;
    } else if (strlen(logFilePath) > 0) {
        logFile = fopen(logFilePath, appendMode ? "a" : "w");
        if (logFile != NULL) {
            openedAppendMode = appendMode;
            return true;
        }
        return false;
    }
    return false;
}

/**
 * Set the log file path name using the following pattern
 *  - Use the module log file if available (unset when first run by ngen), otherwise 
 *  - Use ngen log file if available, otherwise
 *  - Use /ngencerf/data/run-logs/<username>/<module>_<YYMMDDTHHMMSS> if available, othrewise
 *  - Use ~/run-logs/<YYYYMMDD>/<module>_<YYMMDDTHHMMSS>
 *  - Onced opened, save the full log path to the modules log environment variable so
 *    it is only opened once for each ngen run (vs for each catchment)
 */
void SetupLogFile(void) {
    logFilePath[0] = '\0';
    bool appendEntries = true;
    bool moduleLogEnvExists = false;

    const char *envVar = getenv(EV_MODULE_LOGFILEPATH);
    if (envVar && envVar[0] != '\0') {
        strncpy(logFilePath, envVar, sizeof(logFilePath) - 1);
        moduleLogEnvExists = true;
    } else {
        envVar = getenv(EV_NGEN_LOGFILEPATH);
        if (envVar && envVar[0] != '\0') {
            strncpy(logFilePath, envVar, sizeof(logFilePath) - 1);
        } else {
            appendEntries = false;
            char logFileDir[512];

            if (DirectoryExists(LOG_DIR_NGENCERF)) {
                snprintf(logFileDir, sizeof(logFileDir), "%s%s%s", LOG_DIR_NGENCERF, DS, LOG_DIR_DEFAULT);
            } else {
                const char *home = getenv("HOME"); // Get users home directly pathname
                if (home) {
                    snprintf(logFileDir, sizeof(logFileDir), "%s%s%s", home, DS, LOG_DIR_DEFAULT);
                }
                else {
                    snprintf(logFileDir, sizeof(logFileDir), "~%s%s", DS, LOG_DIR_DEFAULT);
                }
            }

            if (CreateDirectory(logFileDir)) {
                const char *envUsername = getenv("USER");
                if (envUsername) {
                    strncat(logFileDir, DS, sizeof(logFileDir) - strlen(logFileDir) - 1);
                    strncat(logFileDir, envUsername, sizeof(logFileDir) - strlen(logFileDir) - 1);
                } else {
                    char dateStr[32];
                    CreateTimestamp(dateStr, sizeof(dateStr), 0, 0);
                    strncat(logFileDir, DS, sizeof(logFileDir) - strlen(logFileDir) - 1);
                    strncat(logFileDir, dateStr, sizeof(logFileDir) - strlen(logFileDir) - 1);
                }

                if (CreateDirectory(logFileDir)) {
                    char timestamp[32];
                    CreateTimestamp(timestamp, sizeof(timestamp), 0, 0);
                    snprintf(logFilePath, sizeof(logFilePath), "%s%s%s_%s.%s",
                             logFileDir, DS, MODULE_NAME, timestamp, LOG_FILE_EXT);
                }
            }
        }
    }

    if (LogFileReady(appendEntries)) {
        if (!moduleLogEnvExists) setenv(EV_MODULE_LOGFILEPATH, logFilePath, 1);
        else {
            printf("%s C Logger: T-Route Python logger already set env var %s=%s\n", MODULE_NAME, EV_MODULE_LOGFILEPATH, logFilePath);
            fflush(stdout); // Force flushing of stdout
        }
        printf("%s C Logger: Opened log file %s in %s mode\n", MODULE_NAME, logFilePath, (openedAppendMode ? "Append" : "Truncate"));
        fflush(stdout); // Force flushing of stdout
    } else {
        printf("[ERROR] %s C Logger: Unable to open log file ", MODULE_NAME);
        if (strlen(logFilePath) > 0) {
            printf("[%s (Perhaps check permissions)\n", logFilePath);
        }
        else printf("\n");
        printf("[WARNING] %s C Logger: Log entries will be written to stdout\n", MODULE_NAME);
        fflush(stdout); // Force flushing of stdout
    }
}

const char* ConvertLogLevelToString(LogLevel level) {
    switch (level) {
        case DEBUG:   return "DEBUG  ";
        case INFO:    return "INFO   ";
        case WARNING: return "WARNING";
        case SEVERE:  return "SEVERE ";
        case FATAL:   return "FATAL  ";
        default:      return "NONE   ";
    }
}

LogLevel ConvertStringToLogLevel(const char* str) {
    char trimmedStr[20] = {0};
    TrimString(str, trimmedStr, sizeof(trimmedStr));
    if (strcasecmp(trimmedStr, "DEBUG")   == 0) return DEBUG;
    if (strcasecmp(trimmedStr, "INFO")    == 0) return INFO;
    if (strcasecmp(trimmedStr, "WARNING") == 0) return WARNING;
    if (strcasecmp(trimmedStr, "SEVERE")  == 0) return SEVERE;
    if (strcasecmp(trimmedStr, "FATAL")   == 0) return FATAL;
    return NONE;
}

void SetLoggingFlag(void) {
    const char* ewtsStr = getenv(EV_EWTS_LOGGING);
    if (ewtsStr && ewtsStr[0] != '\0') {
        char trimmedStr[20] = {0};
        TrimString(ewtsStr, trimmedStr, sizeof(trimmedStr));
        loggingEnabled = ((strcasecmp(trimmedStr, "ENABLED") == 0))? true:false;
    }
    printf("%s C Logger: Logging %s\n", MODULE_NAME, ((loggingEnabled)?"ENABLED":"DISABLED"));
    fflush(stdout); // Force flushing of stdout
}

void SetLogLevel(void) {
    const char* envLogLevel = getenv(EV_MODULE_LOGLEVEL);
    if (envLogLevel && envLogLevel[0] != '\0') {
        logLevel = ConvertStringToLogLevel(envLogLevel);
    }
    char llStr[10];
    char msg[100];
    TrimString(ConvertLogLevelToString(logLevel), llStr, sizeof(llStr));
    snprintf(msg, sizeof(msg), "C Logger: Log level set to %s\n", llStr);
    printf("%s %s\n", MODULE_NAME, msg);
    fflush(stdout); // Force flushing of stdout
    LogLevel saveLevel = logLevel;
    logLevel = INFO; // Ensure this INFO message is always logged
    Log(logLevel, msg);
    logLevel = saveLevel;
}

void SetLogPreferences(void) {

    if (!loggerInitialized) {
        loggerInitialized = true; // Only call this once
        
        SetLoggingFlag(); // Based on the environment variable
        if (loggingEnabled) {
            SetLogModuleName(); // Set the module name used in log entries
            SetupLogFile();
            SetLogLevel();
            // FOR TESTING ONLY! Uncomment next line to generate test log entries
            // WriteLogTestMsgs();
        }
    }
}

void TrimToOneNewline(char *str, size_t max_len) {
    size_t len = strlen(str);

    // Strip trailing whitespace including \n, \r, spaces, tabs
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        str[--len] = '\0';
    }

    // Ensure room to add a newline
    if (len + 1 < max_len) {
        str[len] = '\n';
        str[len + 1] = '\0';
    }
}

LogLevel GetLogLevel(void) {
    return logLevel;
}

bool IsLoggingEnabled(void) {
    return loggingEnabled;
}

void Log(LogLevel messageLevel, const char* message, ...) {
    if (!loggerInitialized) SetLogPreferences(); // Initialize logger on first call

    if (loggingEnabled && (messageLevel >= logLevel)) {

        va_list arglist;
        va_start(arglist, message);
        va_list arglist_copy;
        va_copy(arglist_copy, arglist);

        int length = vsnprintf(NULL, 0, message, arglist_copy);
        va_end(arglist_copy);

        char *buffer = malloc(length + 1);
        if (!buffer) return; // Always good to check

        vsnprintf(buffer, length + 1, message, arglist);
        va_end(arglist);

        char timestamp[32];
        CreateTimestamp(timestamp, sizeof(timestamp), 1, 1);

        char logPrefix[128];
        snprintf(logPrefix, sizeof(logPrefix), "%s %s %s", timestamp, moduleName, ConvertLogLevelToString(messageLevel));

        TrimToOneNewline(buffer, sizeof(buffer));
        char *line = strtok(buffer, "\n");
        if (LogFileReady(true)) {
            while (line != NULL) {
                fprintf(logFile, "%s %s\n", logPrefix, line);
                line = strtok(NULL, "\n");
            }
            fflush(logFile);
        } else {
            while (line != NULL) {
                printf("%s %s\n", logPrefix, line);
                line = strtok(NULL, "\n");
            }
            fflush(stdout); // Force flushing of stdout
        }

        free(buffer);
    }
}

void WriteLogTestMsgs() {

    LogLevel savedLogLevel = logLevel;
    
    setenv(EV_MODULE_LOGLEVEL, "SEVERE", 1);  Log(SEVERE, "Sample Log for LogLevel::SEVERE");
    setenv(EV_MODULE_LOGLEVEL, "FATAL", 1);   Log(FATAL, "Sample Log for LogLevel::FATAL");
    setenv(EV_MODULE_LOGLEVEL, "WARNING", 1); Log(WARNING, "Sample Log for LogLevel::WARNING");
    setenv(EV_MODULE_LOGLEVEL, "INFO", 1);    Log(INFO, "Sample Log for LogLevel::INFO");
    setenv(EV_MODULE_LOGLEVEL, "DEBUG", 1);   Log(DEBUG, "Sample Log for LogLevel::DEBUG");
    
    const char* multiline_log = 
        "First line of multiline log:\n"
        "Second line of multiline log\n"
        "Third line of multiline log\n"
        "Fourth line of multiline log";
    Log(DEBUG, multiline_log);
    
    setenv(EV_MODULE_LOGLEVEL, ConvertLogLevelToString(savedLogLevel), 1); 
}
