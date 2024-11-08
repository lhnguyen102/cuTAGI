#pragma once
#include <stdexcept>
#include <string>

// Log Levels
enum class LogLevel { INFO, WARNING, ERROR, DEBUG };

class Logger {
   public:
    static void log_message(LogLevel level, const char* file, int line,
                            const std::string& message);

   private:
    static std::string current_utc_time();

    static std::string log_level_to_string(LogLevel level);
};

// Macro to simplify logging with automatic file and line number
#define LOG(level, message) \
    Logger::log_message(level, __FILE__, __LINE__, message)
