#include "custom_logger.h"

#include <ctime>
#include <iostream>
#include <stdexcept>

std::string Logger::log_level_to_string(LogLevel level)
/*Convert LogLevel to string*/
{
    switch (level) {
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::DEBUG:
            return "DEBUG";
        default:
            return "UNKNOWN";
    }
}

std::string Logger::current_utc_time()
/* Get utc time for logging*/
{
    time_t now = time(0);
    tm* utc_time = gmtime(&now);
    char buffer[80];
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S UTC", utc_time);
    return std::string(buffer);
}

void Logger::log_message(LogLevel level, const char* file, int line,
                         const std::string& message)
/*Print the log message alongside with file name and line*/
{
    std::cout << "[" << current_utc_time() << "] " << log_level_to_string(level)
              << " " << file << ":" << line << " - " << message << std::endl;

    if (level == LogLevel::ERROR) {
        throw std::runtime_error(message);
    }
}
