#include "../include/scope_logger.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>

namespace debug {
    // Initialize static members
    std::mutex ScopeLogger::log_mutex;
    thread_local int ScopeLogger::indent_level = 0;
    std::string ScopeLogger::log_path;

    void ScopeLogger::init(const std::string& log_directory) {
        if (!scope_logging_enabled) return;  // Skip if logging is disabled
        
        std::lock_guard<std::mutex> lock(log_mutex);
        std::filesystem::path dir_path(log_directory);
        if (!std::filesystem::exists(dir_path)) {
            std::filesystem::create_directories(dir_path);
        }
        log_path = (dir_path / "scope_trace.log").string();
        
        // Clear the log file at initialization
        std::ofstream log_file(log_path, std::ios::trunc);
        if (log_file.is_open()) {
            log_file << "=== Scope Tracing Log Started ===\n" << std::endl;
        }
    }

    ScopeLogger::ScopeLogger(const std::string& scope, const std::string& file, int line) 
        : scope_name(scope), file_name(file), line_number(line) {
        if (!scope_logging_enabled) return;  // Skip if logging is disabled
        
        std::lock_guard<std::mutex> lock(log_mutex);
        log_entry("Entering");
        indent_level++;
    }

    ScopeLogger::~ScopeLogger() {
        if (!scope_logging_enabled || scope_name.empty()) return;  // Skip if logging is disabled
        
        std::lock_guard<std::mutex> lock(log_mutex);
        indent_level--;
        log_entry("Exiting");
    }

    void ScopeLogger::log_entry(const std::string& action) {
        if (!scope_logging_enabled || scope_name.empty()) return;  // Skip if logging is disabled

        if (log_path.empty()) {
            init();  // Initialize with default path if not set
        }
        
        std::ofstream log_file(log_path, std::ios::app);
        if (log_file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            auto now_tm = *std::localtime(&now_time_t);
            
            std::string indent(indent_level * 2, ' ');
            log_file << std::put_time(&now_tm, "%H:%M:%S") << " "
                    << indent << action << " " << scope_name 
                    << " [" << file_name << ":" << line_number << "]"
                    << " (Thread: " << std::this_thread::get_id() << ")"
                    << std::endl;
        }
    }
}
