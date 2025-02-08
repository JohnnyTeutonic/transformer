#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <filesystem>

class ScopeLogger {
private:
    static std::mutex log_mutex;
    static thread_local int indent_level;
    static std::string log_path;
    std::string scope_name;
    std::string file_name;
    int line_number;

public:
    // Initialize log directory and file
    static void init(const std::string& log_directory = "../build");

    // Constructor to enter a scope
    ScopeLogger(const std::string& scope, const std::string& file, int line);

    // Destructor to exit a scope
    ~ScopeLogger();

private:
    void log_entry(const std::string& action);
};

// Define a macro for easy usage
#define SCOPE_LOG() ScopeLogger scope_logger(__FUNCTION__, __FILE__, __LINE__) 