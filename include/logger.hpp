#pragma once
#include <string>
#include <fstream>

enum class LogLevel {
    INFO = 0,
    DEBUG = 1,
    WARNING = 2,
    ERROR = 3
};

class Logger {
private:
    static Logger* instance;
    std::ofstream log_file;
    bool logging_enabled;
    LogLevel current_level;
    static std::streambuf* cout_buffer;  // Store original cout buffer
    
    Logger();  // Private constructor for singleton
    
public:
    static Logger& getInstance();
    void log(const std::string& message);
    void log(const std::string& message, LogLevel level);
    void setLogLevel(LogLevel level);
    LogLevel getLogLevel() const;
    void enableLogging();
    void disableLogging();
    ~Logger();
};