#include "../include/logger.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

Logger* Logger::instance = nullptr;
// Save the original cout buffer
std::streambuf* Logger::cout_buffer = nullptr;

Logger::Logger() : logging_enabled(true), current_level(LogLevel::INFO) {
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&time_t_now), "%d%m%Y_%H%M%S");
    
    // Create logs directory if it doesn't exist
    std::filesystem::create_directories("logs");
    
    // Create log filename with datetime
    std::string log_filename = "logs/transformer_" + datetime.str() + ".log";
    
    log_file.open(log_filename, std::ios::out | std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << log_filename << std::endl;
    }
    else {
        // Log the start time
        log_file << "=== Log started at: " << std::put_time(std::localtime(&time_t_now), "%d-%m-%Y %H:%M:%S") << " ===" << std::endl;
    }
}

Logger& Logger::getInstance() {
    if (instance == nullptr) {
        instance = new Logger();
    }
    return *instance;
}

void Logger::log(const std::string& message) {
    log(message, LogLevel::INFO);
}

void Logger::log(const std::string& message, LogLevel level) {
    if (!logging_enabled || level < current_level) return;
    
    const char* level_str;
    switch(level) {
        case LogLevel::INFO:    level_str = "INFO"; break;
        case LogLevel::DEBUG:   level_str = "DEBUG"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::ERROR:   level_str = "ERROR"; break;
    }
    
    if (log_file.is_open()) {
        log_file << "[" << level_str << "] " << message << std::endl;
        log_file.flush();
    } else {
        // Temporary fallback to stderr if file isn't open
        std::cerr << "Log file not open, writing to stderr: [" << level_str << "] " << message << std::endl;
    }
}

void Logger::setLogLevel(LogLevel level) {
    current_level = level;
}

LogLevel Logger::getLogLevel() const {
    return current_level;
}

void Logger::enableLogging() {
    logging_enabled = true;
    if (log_file.is_open()) {
        // Save cout's buffer before redirect
        cout_buffer = std::cout.rdbuf();
        // Redirect cout to log file
        std::cout.rdbuf(log_file.rdbuf());
    }
}

void Logger::disableLogging() {
    logging_enabled = false;
    // Restore cout's original buffer if we saved it
    if (cout_buffer != nullptr) {
        std::cout.rdbuf(cout_buffer);
        cout_buffer = nullptr;
    }
}

Logger::~Logger() {
    // Restore cout's original buffer if we saved it
    if (cout_buffer != nullptr) {
        std::cout.rdbuf(cout_buffer);
    }
    if (log_file.is_open()) {
        log_file.close();
    }
    if (instance) {
        delete instance;
        instance = nullptr;
    }
}