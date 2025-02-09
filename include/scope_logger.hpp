#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <filesystem>

namespace debug {
    // Forward declare scope_logging_enabled from debug namespace
    extern bool scope_logging_enabled;

    class ScopeLogger {
    private:
        static std::mutex log_mutex;
        static thread_local int indent_level;
        static std::string log_path;
        std::string scope_name;
        std::string file_name;
        int line_number;

    public:
        // Default constructor for disabled case
        ScopeLogger() : scope_name(""), file_name(""), line_number(0) {}

        // Initialize log directory and file
        static void init(const std::string& log_directory = "../build");

        // Constructor to enter a scope
        ScopeLogger(const std::string& scope, const std::string& file, int line);

        // Destructor to exit a scope
        ~ScopeLogger();

        // Public method to clear log path
        static void clear_log_path() { 
            if (!scope_logging_enabled) return;
            log_path.clear(); 
        }

    private:
        void log_entry(const std::string& action);
    };
}

// Define a macro for easy usage that checks if scope logging is enabled
#define SCOPE_LOG() debug::ScopeLogger scope_logger_instance = debug::ScopeLogger(__FUNCTION__, __FILE__, __LINE__) 