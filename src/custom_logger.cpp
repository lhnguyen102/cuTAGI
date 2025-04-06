#include "custom_logger.h"

#include <ctime>
#include <iostream>
#include <stdexcept>

// ANSI escape codes for colors and formatting
namespace {
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* BLUE = "\033[34m";
}  // namespace

std::string Logger::log_level_to_string(LogLevel level)
/*Convert LogLevel to string with color*/
{
    switch (level) {
        case LogLevel::INFO:
            return std::string(BOLD) + GREEN + "[INFO]" + RESET;
        case LogLevel::WARNING:
            return std::string(BOLD) + YELLOW + "[WARNING]" + RESET;
        case LogLevel::ERROR:
            return std::string(BOLD) + RED + "[ERROR]" + RESET;
        case LogLevel::DEBUG:
            return std::string(BOLD) + BLUE + "[DEBUG]" + RESET;
        default:
            return "[UNKNOWN]";
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
    if (level == LogLevel::INFO) {
        std::cout << "[" << current_utc_time() << "] "
                  << log_level_to_string(level) << " - " << message
                  << std::endl;
    } else {
        std::cout << "[" << current_utc_time() << "] "
                  << log_level_to_string(level) << " " << file << ":" << line
                  << " - " << message << std::endl;
    }

    if (level == LogLevel::ERROR) {
        throw std::runtime_error(message);
    }
}
