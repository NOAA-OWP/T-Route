#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h> // for variable args: va_list

typedef enum {      
    NONE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    SEVERE = 4,
    FATAL = 5,
} LogLevel;

// Public Methods
void Log(LogLevel messageLevel, const char* message, ...);

#endif // LOGGER_H
