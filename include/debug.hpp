#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <mutex>

namespace debug {
    extern bool verbose_logging;
    extern std::ofstream debug_log;
    extern const std::string log_file;
    extern std::mutex log_mutex;

    void init_logging();
    void log_message(const std::string& message, const std::string& level = "INFO");
    void log_vector(const std::vector<int>& vec, const std::string& name);
} 