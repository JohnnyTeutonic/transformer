#pragma once
#include <algorithm>
#include <string>
#include <fstream>
#include <deque>
#include <vector>
#include <mutex>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

class LossVisualizer {
public:
    static constexpr size_t MAX_HISTORY = 1000;  // Maximum number of data points to keep
    static constexpr size_t TREND_LINE_WIDTH = 50;  // Width of the ASCII trend line
    static constexpr size_t SHORT_WINDOW = 10;   // Short-term moving average window
    static constexpr size_t MEDIUM_WINDOW = 50;  // Medium-term moving average window

    LossVisualizer(const std::string& log_path = "./loss.log") 
        : log_file_path(log_path), current_lr(0.0f) {
        try {
            // Open file in truncate mode (std::ios::trunc) to clear previous contents
            log_file.open(log_file_path, std::ios::out | std::ios::trunc);
            if (!log_file.is_open()) {
                std::cerr << "Warning: Could not open " << log_file_path << ", trying current directory..." << std::endl;
                // Fallback to current directory
                log_file_path = "loss.log";
                log_file.open(log_file_path, std::ios::out | std::ios::trunc);
                if (!log_file.is_open()) {
                    throw std::runtime_error("Failed to open log file in current directory");
                }
            }
            
            // Write initial header with program start time
            log_file << "=== Training Log Started at " << get_timestamp() << " ===\n";
            log_file << "Window Sizes: Short=" << SHORT_WINDOW << ", Medium=" << MEDIUM_WINDOW << "\n";
            log_file << "Max History: " << MAX_HISTORY << " samples\n";
            log_file << "----------------------------------------\n\n";
            log_file.flush();
        } catch (const std::exception& e) {
            std::cerr << "Error initializing loss visualizer: " << e.what() << std::endl;
            throw;
        }
    }

    ~LossVisualizer() {
        if (log_file.is_open()) {
            // Write final summary
            log_file << "\n=== Training Summary ===\n";
            if (!raw_history.empty()) {
                log_file << "Final Statistics:\n";
                log_file << "Raw Loss - Min: " << raw_stats.min 
                        << ", Max: " << raw_stats.max 
                        << ", Final: " << raw_history.back() << "\n";
                log_file << "Smoothed Loss - Min: " << smoothed_stats.min 
                        << ", Max: " << smoothed_stats.max 
                        << ", Final: " << smoothed_history.back() << "\n";
                if (!grad_norm_history.empty()) {
                    log_file << "Gradient Norm - Min: " << grad_stats.min
                            << ", Max: " << grad_stats.max
                            << ", Final: " << grad_norm_history.back() << "\n";
                }
                log_file << "Total Samples: " << raw_history.size() << "\n";
            }
            log_file << "Training Completed at " << get_timestamp() << "\n";
            log_file << "----------------------------------------\n";
            log_file.close();
        }
    }

    void add_loss(float raw_loss, float smoothed_loss, float trend, float grad_norm = 0.0f, float learning_rate = 0.0f) {
        std::lock_guard<std::mutex> lock(mutex);
        current_lr = learning_rate;  // Store current learning rate
        
        // Add to history
        raw_history.push_back(raw_loss);
        smoothed_history.push_back(smoothed_loss);
        trend_history.push_back(trend);
        if (grad_norm > 0.0f) {
            grad_norm_history.push_back(grad_norm);
        }

        // Maintain maximum size
        if (raw_history.size() > MAX_HISTORY) {
            raw_history.pop_front();
            smoothed_history.pop_front();
            trend_history.pop_front();
            if (!grad_norm_history.empty()) {
                grad_norm_history.pop_front();
            }
        }

        // Update statistics
        update_statistics();
        
        // Write to log file
        write_log();
    }

private:
    std::string log_file_path;
    std::ofstream log_file;
    std::mutex mutex;
    float current_lr;  // Add learning rate tracking

    std::deque<float> raw_history;
    std::deque<float> smoothed_history;
    std::deque<float> trend_history;
    std::deque<float> grad_norm_history;  // Store gradient norms

    // Statistics
    struct Stats {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        float mean = 0.0f;
        float std_dev = 0.0f;
        float short_ma = 0.0f;  // Short-term moving average
        float medium_ma = 0.0f; // Medium-term moving average
        char trend_indicator = '-';  // '+' for up, '-' for stable, 'v' for down
    };

    Stats raw_stats;
    Stats smoothed_stats;
    Stats trend_stats;
    Stats grad_stats;  // Statistics for gradients

    void update_statistics() {
        raw_stats = calculate_stats(raw_history);
        smoothed_stats = calculate_stats(smoothed_history);
        trend_stats = calculate_stats(trend_history);
        if (!grad_norm_history.empty()) {
            grad_stats = calculate_stats(grad_norm_history);
        }
    }

    Stats calculate_stats(const std::deque<float>& data) {
        Stats stats;
        if (data.empty()) return stats;

        // Calculate min, max, and mean
        float sum = 0.0f;
        for (float value : data) {
            stats.min = std::min(stats.min, value);
            stats.max = std::max(stats.max, value);
            sum += value;
        }
        stats.mean = sum / data.size();

        // Calculate standard deviation
        float sum_sq_diff = 0.0f;
        for (float value : data) {
            float diff = value - stats.mean;
            sum_sq_diff += diff * diff;
        }
        stats.std_dev = std::sqrt(sum_sq_diff / data.size());

        // Moving averages
        size_t short_window = std::min(SHORT_WINDOW, data.size());
        size_t medium_window = std::min(MEDIUM_WINDOW, data.size());
        
        if (short_window > 0) {
            float short_sum = 0.0f;
            for (size_t i = data.size() - short_window; i < data.size(); ++i) {
                short_sum += data[i];
            }
            stats.short_ma = short_sum / short_window;
        }

        if (medium_window > 0) {
            float medium_sum = 0.0f;
            for (size_t i = data.size() - medium_window; i < data.size(); ++i) {
                medium_sum += data[i];
            }
            stats.medium_ma = medium_sum / medium_window;
        }

        // Trend indicator using ASCII characters
        if (data.size() >= 2) {
            float recent_avg = stats.short_ma;
            float older_avg = stats.medium_ma;
            float threshold = 0.001f;  // Threshold for considering a change significant
            
            if (std::abs(recent_avg - older_avg) < threshold) {
                stats.trend_indicator = '-';  // Stable
            } else if (recent_avg < older_avg) {
                stats.trend_indicator = 'v';  // Down
            } else {
                stats.trend_indicator = '+';  // Up
            }
        }

        return stats;
    }

    std::string create_trend_line(const std::deque<float>& data, float min_val, float max_val) {
        if (data.empty()) return std::string(TREND_LINE_WIDTH, '-');

        std::string line(TREND_LINE_WIDTH, ' ');
        float range = max_val - min_val;
        if (range == 0) range = 1.0f;  // Prevent division by zero

        // Use only the most recent values that fit in the trend line width
        size_t start_idx = data.size() > TREND_LINE_WIDTH ? data.size() - TREND_LINE_WIDTH : 0;
        for (size_t i = 0; i < std::min(TREND_LINE_WIDTH, data.size()); ++i) {
            size_t data_idx = start_idx + i;
            float normalized = (data[data_idx] - min_val) / range;
            line[i] = get_trend_char(normalized);
        }

        return line;
    }

    char get_trend_char(float normalized) {
        // Convert normalized value [0,1] to ASCII art character
        // Using a carefully selected set of ASCII characters for clear visualization
        // Ordered from lowest to highest intensity
        const char trend_chars[] = " .,:-=+*#@";  // space is lowest, @ is highest
        int index = static_cast<int>(normalized * (sizeof(trend_chars) - 2));
        index = std::clamp(index, 0, static_cast<int>(sizeof(trend_chars) - 2));
        return trend_chars[index];
    }

    void write_log() {
        // Write current timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        log_file << "\n=== Update at " << std::ctime(&time);

        // Add learning rate to log
        log_file << "Learning Rate: " << std::scientific << std::setprecision(3) << current_lr << "\n\n";

        // Raw Loss Section with trend indicator and moving averages
        log_file << "Raw Loss " << raw_stats.trend_indicator << ":\n";
        log_file << create_trend_line(raw_history, raw_stats.min, raw_stats.max) << "\n";
        log_file << "Min: " << std::fixed << std::setprecision(6) << raw_stats.min 
                << " | Max: " << raw_stats.max 
                << " | Mean: " << raw_stats.mean 
                << " | StdDev: " << raw_stats.std_dev << "\n";
        log_file << "Short MA: " << raw_stats.short_ma 
                << " | Medium MA: " << raw_stats.medium_ma << "\n\n";

        // Smoothed Loss Section
        log_file << "Smoothed Loss " << smoothed_stats.trend_indicator << ":\n";
        log_file << create_trend_line(smoothed_history, smoothed_stats.min, smoothed_stats.max) << "\n";
        log_file << "Min: " << smoothed_stats.min 
                << " | Max: " << smoothed_stats.max 
                << " | Mean: " << smoothed_stats.mean 
                << " | StdDev: " << smoothed_stats.std_dev << "\n";
        log_file << "Short MA: " << smoothed_stats.short_ma 
                << " | Medium MA: " << smoothed_stats.medium_ma << "\n\n";

        // Gradient Section (if available)
        if (!grad_norm_history.empty()) {
            log_file << "Gradient Norm " << grad_stats.trend_indicator << ":\n";
            log_file << create_trend_line(grad_norm_history, grad_stats.min, grad_stats.max) << "\n";
            log_file << "Min: " << grad_stats.min 
                    << " | Max: " << grad_stats.max 
                    << " | Mean: " << grad_stats.mean 
                    << " | StdDev: " << grad_stats.std_dev << "\n";
            log_file << "Short MA: " << grad_stats.short_ma 
                    << " | Medium MA: " << grad_stats.medium_ma << "\n\n";
        }

        // Loss Trend Section
        log_file << "Loss Trend " << trend_stats.trend_indicator << " (recent/overall):\n";
        log_file << create_trend_line(trend_history, trend_stats.min, trend_stats.max) << "\n";
        log_file << "Min: " << trend_stats.min 
                << " | Max: " << trend_stats.max 
                << " | Mean: " << trend_stats.mean 
                << " | StdDev: " << trend_stats.std_dev << "\n";
        log_file << "Short MA: " << trend_stats.short_ma 
                << " | Medium MA: " << trend_stats.medium_ma << "\n\n";

        // Sample counts
        log_file << "History size: " << raw_history.size() << "/" << MAX_HISTORY << " samples\n";
        log_file << "----------------------------------------\n";

        log_file.flush();
    }

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time);
        // Remove newline that ctime adds
        if (!timestamp.empty() && timestamp[timestamp.length()-1] == '\n') {
            timestamp.erase(timestamp.length()-1);
        }
        return timestamp;
    }
}; 