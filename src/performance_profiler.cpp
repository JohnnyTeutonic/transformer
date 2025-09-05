#include "../include/performance_profiler.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <nvml.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#endif

namespace profiling {

PerformanceProfiler::PerformanceProfiler(const ProfilerConfig& config)
    : config_(config), profiling_active_(false) {
    
    std::cout << "PerformanceProfiler initialized with:" << std::endl;
    std::cout << "- Sampling interval: " << config_.sampling_interval_ms << "ms" << std::endl;
    std::cout << "- History size: " << config_.history_size << std::endl;
    std::cout << "- GPU monitoring: " << (config_.enable_gpu_monitoring ? "enabled" : "disabled") << std::endl;
    std::cout << "- Network monitoring: " << (config_.enable_network_monitoring ? "enabled" : "disabled") << std::endl;
    
    // Initialize NVML for GPU monitoring
#ifdef CUDA_AVAILABLE
    if (config_.enable_gpu_monitoring) {
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS) {
            nvml_initialized_ = true;
            std::cout << "NVML initialized successfully for GPU monitoring" << std::endl;
        } else {
            std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        }
    }
#endif
    
    // Initialize performance counters
    reset_counters();
}

PerformanceProfiler::~PerformanceProfiler() {
    stop_profiling();
    
#ifdef CUDA_AVAILABLE
    if (nvml_initialized_) {
        nvmlShutdown();
    }
#endif
}

bool PerformanceProfiler::start_profiling() {
    if (profiling_active_.load()) {
        std::cout << "Performance profiling already active" << std::endl;
        return true;
    }
    
    std::cout << "Starting performance profiling..." << std::endl;
    
    profiling_active_.store(true);
    
    // Start profiling threads
    cpu_monitor_thread_ = std::thread(&PerformanceProfiler::cpu_monitoring_thread, this);
    
    if (config_.enable_gpu_monitoring) {
        gpu_monitor_thread_ = std::thread(&PerformanceProfiler::gpu_monitoring_thread, this);
    }
    
    if (config_.enable_network_monitoring) {
        network_monitor_thread_ = std::thread(&PerformanceProfiler::network_monitoring_thread, this);
    }
    
    memory_monitor_thread_ = std::thread(&PerformanceProfiler::memory_monitoring_thread, this);
    
    std::cout << "Performance profiling started" << std::endl;
    return true;
}

void PerformanceProfiler::stop_profiling() {
    if (!profiling_active_.load()) {
        return;
    }
    
    std::cout << "Stopping performance profiling..." << std::endl;
    
    profiling_active_.store(false);
    
    // Notify all condition variables
    profiling_cv_.notify_all();
    
    // Join threads
    if (cpu_monitor_thread_.joinable()) {
        cpu_monitor_thread_.join();
    }
    if (gpu_monitor_thread_.joinable()) {
        gpu_monitor_thread_.join();
    }
    if (network_monitor_thread_.joinable()) {
        network_monitor_thread_.join();
    }
    if (memory_monitor_thread_.joinable()) {
        memory_monitor_thread_.join();
    }
    
    std::cout << "Performance profiling stopped" << std::endl;
}

void PerformanceProfiler::record_training_step(const TrainingStepMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    training_history_.push_back(metrics);
    
    // Keep only recent history
    if (training_history_.size() > config_.history_size) {
        training_history_.erase(training_history_.begin(), 
                               training_history_.begin() + (training_history_.size() - config_.history_size));
    }
    
    // Update running statistics
    update_training_statistics(metrics);
}

void PerformanceProfiler::record_network_operation(const NetworkOperationMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    network_history_.push_back(metrics);
    
    // Keep only recent history
    if (network_history_.size() > config_.history_size) {
        network_history_.erase(network_history_.begin(),
                              network_history_.begin() + (network_history_.size() - config_.history_size));
    }
    
    // Update network statistics
    update_network_statistics(metrics);
}

SystemCapabilities PerformanceProfiler::get_system_capabilities() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    SystemCapabilities caps;
    
    // CPU capabilities
    caps.cpu_cores = get_cpu_core_count();
    caps.cpu_frequency_mhz = get_cpu_frequency();
    
    // Memory capabilities
    caps.total_memory_gb = get_total_memory_gb();
    caps.available_memory_gb = get_available_memory_gb();
    
    // GPU capabilities
#ifdef CUDA_AVAILABLE
    if (nvml_initialized_) {
        caps.gpu_count = get_gpu_count();
        if (caps.gpu_count > 0) {
            caps.gpu_memory_gb = get_gpu_memory_gb(0);  // Primary GPU
            caps.gpu_compute_capability = get_gpu_compute_capability(0);
        }
    }
#endif
    
    // Network capabilities (estimated from recent measurements)
    if (!network_history_.empty()) {
        auto recent_start = network_history_.end() - std::min(static_cast<size_t>(10), network_history_.size());
        
        float total_bandwidth = 0.0f;
        float total_latency = 0.0f;
        size_t count = 0;
        
        for (auto it = recent_start; it != network_history_.end(); ++it) {
            total_bandwidth += it->bandwidth_mbps;
            total_latency += it->latency_ms;
            count++;
        }
        
        if (count > 0) {
            caps.network_bandwidth_mbps = total_bandwidth / count;
            caps.network_latency_ms = total_latency / count;
        }
    }
    
    return caps;
}

PerformanceMetrics PerformanceProfiler::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

PerformanceProfile PerformanceProfiler::get_performance_profile() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    PerformanceProfile profile;
    profile.timestamp = std::chrono::steady_clock::now();
    profile.system_capabilities = get_system_capabilities();
    profile.current_metrics = current_metrics_;
    
    // Calculate performance scores
    profile.compute_score = calculate_compute_score();
    profile.memory_score = calculate_memory_score();
    profile.network_score = calculate_network_score();
    profile.overall_score = (profile.compute_score + profile.memory_score + profile.network_score) / 3.0f;
    
    // Add recent training performance
    if (!training_history_.empty()) {
        auto recent_start = training_history_.end() - std::min(static_cast<size_t>(10), training_history_.size());
        
        float total_throughput = 0.0f;
        size_t count = 0;
        
        for (auto it = recent_start; it != training_history_.end(); ++it) {
            total_throughput += it->samples_per_second;
            count++;
        }
        
        profile.recent_training_throughput = count > 0 ? total_throughput / count : 0.0f;
    }
    
    return profile;
}

void PerformanceProfiler::cpu_monitoring_thread() {
    std::cout << "CPU monitoring thread started" << std::endl;
    
    while (profiling_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Sample CPU metrics
        CPUMetrics cpu_metrics = sample_cpu_metrics();
        
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.cpu = cpu_metrics;
            cpu_history_.push_back({std::chrono::steady_clock::now(), cpu_metrics});
            
            // Keep only recent history
            if (cpu_history_.size() > config_.history_size) {
                cpu_history_.erase(cpu_history_.begin());
            }
        }
        
        // Sleep until next sampling interval
        std::unique_lock<std::mutex> lock(profiling_mutex_);
        profiling_cv_.wait_for(lock, std::chrono::milliseconds(config_.sampling_interval_ms),
                              [this] { return !profiling_active_.load(); });
    }
    
    std::cout << "CPU monitoring thread stopped" << std::endl;
}

void PerformanceProfiler::gpu_monitoring_thread() {
    std::cout << "GPU monitoring thread started" << std::endl;
    
    while (profiling_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Sample GPU metrics
        GPUMetrics gpu_metrics = sample_gpu_metrics();
        
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.gpu = gpu_metrics;
            gpu_history_.push_back({std::chrono::steady_clock::now(), gpu_metrics});
            
            // Keep only recent history
            if (gpu_history_.size() > config_.history_size) {
                gpu_history_.erase(gpu_history_.begin());
            }
        }
        
        // Sleep until next sampling interval
        std::unique_lock<std::mutex> lock(profiling_mutex_);
        profiling_cv_.wait_for(lock, std::chrono::milliseconds(config_.sampling_interval_ms * 2), // GPU sampling less frequent
                              [this] { return !profiling_active_.load(); });
    }
    
    std::cout << "GPU monitoring thread stopped" << std::endl;
}

void PerformanceProfiler::memory_monitoring_thread() {
    std::cout << "Memory monitoring thread started" << std::endl;
    
    while (profiling_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Sample memory metrics
        MemoryMetrics memory_metrics = sample_memory_metrics();
        
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.memory = memory_metrics;
            memory_history_.push_back({std::chrono::steady_clock::now(), memory_metrics});
            
            // Keep only recent history
            if (memory_history_.size() > config_.history_size) {
                memory_history_.erase(memory_history_.begin());
            }
        }
        
        // Sleep until next sampling interval
        std::unique_lock<std::mutex> lock(profiling_mutex_);
        profiling_cv_.wait_for(lock, std::chrono::milliseconds(config_.sampling_interval_ms),
                              [this] { return !profiling_active_.load(); });
    }
    
    std::cout << "Memory monitoring thread stopped" << std::endl;
}

void PerformanceProfiler::network_monitoring_thread() {
    std::cout << "Network monitoring thread started" << std::endl;
    
    while (profiling_active_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Sample network metrics
        NetworkMetrics network_metrics = sample_network_metrics();
        
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.network = network_metrics;
            network_metrics_history_.push_back({std::chrono::steady_clock::now(), network_metrics});
            
            // Keep only recent history
            if (network_metrics_history_.size() > config_.history_size) {
                network_metrics_history_.erase(network_metrics_history_.begin());
            }
        }
        
        // Sleep until next sampling interval
        std::unique_lock<std::mutex> lock(profiling_mutex_);
        profiling_cv_.wait_for(lock, std::chrono::milliseconds(config_.sampling_interval_ms * 3), // Network sampling less frequent
                              [this] { return !profiling_active_.load(); });
    }
    
    std::cout << "Network monitoring thread stopped" << std::endl;
}

CPUMetrics PerformanceProfiler::sample_cpu_metrics() {
    CPUMetrics metrics;
    
#ifdef _WIN32
    // Windows CPU sampling
    FILETIME idle_time, kernel_time, user_time;
    if (GetSystemTimes(&idle_time, &kernel_time, &user_time)) {
        // Convert to usage percentage (simplified)
        metrics.usage_percent = 50.0f;  // Placeholder
    }
#elif defined(__linux__)
    // Linux CPU sampling from /proc/stat
    std::ifstream stat_file("/proc/stat");
    if (stat_file.is_open()) {
        std::string line;
        std::getline(stat_file, line);
        
        // Parse CPU times (simplified)
        metrics.usage_percent = 45.0f;  // Placeholder
    }
#endif
    
    metrics.core_count = get_cpu_core_count();
    metrics.frequency_mhz = get_cpu_frequency();
    metrics.temperature_celsius = get_cpu_temperature();
    
    return metrics;
}

GPUMetrics PerformanceProfiler::sample_gpu_metrics() {
    GPUMetrics metrics;
    
#ifdef CUDA_AVAILABLE
    if (nvml_initialized_) {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
        
        if (result == NVML_SUCCESS) {
            // GPU utilization
            nvmlUtilization_t utilization;
            result = nvmlDeviceGetUtilizationRates(device, &utilization);
            if (result == NVML_SUCCESS) {
                metrics.usage_percent = static_cast<float>(utilization.gpu);
                metrics.memory_usage_percent = static_cast<float>(utilization.memory);
            }
            
            // GPU memory
            nvmlMemory_t memory_info;
            result = nvmlDeviceGetMemoryInfo(device, &memory_info);
            if (result == NVML_SUCCESS) {
                metrics.memory_used_gb = static_cast<float>(memory_info.used) / (1024.0f * 1024.0f * 1024.0f);
                metrics.memory_total_gb = static_cast<float>(memory_info.total) / (1024.0f * 1024.0f * 1024.0f);
            }
            
            // GPU temperature
            unsigned int temperature;
            result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
            if (result == NVML_SUCCESS) {
                metrics.temperature_celsius = static_cast<float>(temperature);
            }
            
            // GPU power
            unsigned int power;
            result = nvmlDeviceGetPowerUsage(device, &power);
            if (result == NVML_SUCCESS) {
                metrics.power_usage_watts = static_cast<float>(power) / 1000.0f;
            }
        }
    }
#endif
    
    return metrics;
}

MemoryMetrics PerformanceProfiler::sample_memory_metrics() {
    MemoryMetrics metrics;
    
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        metrics.total_gb = static_cast<float>(mem_status.ullTotalPhys) / (1024.0f * 1024.0f * 1024.0f);
        metrics.available_gb = static_cast<float>(mem_status.ullAvailPhys) / (1024.0f * 1024.0f * 1024.0f);
        metrics.used_gb = metrics.total_gb - metrics.available_gb;
        metrics.usage_percent = (metrics.used_gb / metrics.total_gb) * 100.0f;
    }
#elif defined(__linux__)
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        metrics.total_gb = static_cast<float>(sys_info.totalram * sys_info.mem_unit) / (1024.0f * 1024.0f * 1024.0f);
        metrics.available_gb = static_cast<float>(sys_info.freeram * sys_info.mem_unit) / (1024.0f * 1024.0f * 1024.0f);
        metrics.used_gb = metrics.total_gb - metrics.available_gb;
        metrics.usage_percent = (metrics.used_gb / metrics.total_gb) * 100.0f;
    }
#endif
    
    return metrics;
}

NetworkMetrics PerformanceProfiler::sample_network_metrics() {
    NetworkMetrics metrics;
    
    // Sample network interface statistics
    // This is a simplified implementation - real implementation would
    // read from /proc/net/dev on Linux or use Windows APIs
    
    metrics.bytes_sent_per_sec = 1024.0f * 1024.0f;  // Placeholder: 1 MB/s
    metrics.bytes_received_per_sec = 2048.0f * 1024.0f;  // Placeholder: 2 MB/s
    metrics.packets_sent_per_sec = 1000.0f;
    metrics.packets_received_per_sec = 1500.0f;
    metrics.latency_ms = 25.0f;  // Placeholder
    
    return metrics;
}

uint32_t PerformanceProfiler::get_cpu_core_count() const {
#ifdef _WIN32
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    return sys_info.dwNumberOfProcessors;
#else
    return static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

float PerformanceProfiler::get_cpu_frequency() const {
    // Simplified - would need platform-specific implementation
    return 3000.0f;  // Placeholder: 3 GHz
}

float PerformanceProfiler::get_cpu_temperature() const {
    // Platform-specific temperature reading would go here
    return 65.0f;  // Placeholder
}

float PerformanceProfiler::get_total_memory_gb() const {
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        return static_cast<float>(mem_status.ullTotalPhys) / (1024.0f * 1024.0f * 1024.0f);
    }
#elif defined(__linux__)
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        return static_cast<float>(sys_info.totalram * sys_info.mem_unit) / (1024.0f * 1024.0f * 1024.0f);
    }
#endif
    return 0.0f;
}

float PerformanceProfiler::get_available_memory_gb() const {
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        return static_cast<float>(mem_status.ullAvailPhys) / (1024.0f * 1024.0f * 1024.0f);
    }
#elif defined(__linux__)
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        return static_cast<float>(sys_info.freeram * sys_info.mem_unit) / (1024.0f * 1024.0f * 1024.0f);
    }
#endif
    return 0.0f;
}

uint32_t PerformanceProfiler::get_gpu_count() const {
#ifdef CUDA_AVAILABLE
    if (nvml_initialized_) {
        unsigned int device_count;
        nvmlReturn_t result = nvmlDeviceGetCount(&device_count);
        if (result == NVML_SUCCESS) {
            return device_count;
        }
    }
#endif
    return 0;
}

float PerformanceProfiler::get_gpu_memory_gb(uint32_t gpu_index) const {
#ifdef CUDA_AVAILABLE
    if (nvml_initialized_) {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(gpu_index, &device);
        if (result == NVML_SUCCESS) {
            nvmlMemory_t memory_info;
            result = nvmlDeviceGetMemoryInfo(device, &memory_info);
            if (result == NVML_SUCCESS) {
                return static_cast<float>(memory_info.total) / (1024.0f * 1024.0f * 1024.0f);
            }
        }
    }
#endif
    return 0.0f;
}

std::string PerformanceProfiler::get_gpu_compute_capability(uint32_t gpu_index) const {
#ifdef CUDA_AVAILABLE
    if (nvml_initialized_) {
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(gpu_index, &device);
        if (result == NVML_SUCCESS) {
            int major, minor;
            result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
            if (result == NVML_SUCCESS) {
                return std::to_string(major) + "." + std::to_string(minor);
            }
        }
    }
#endif
    return "Unknown";
}

float PerformanceProfiler::calculate_compute_score() const {
    float score = 0.0f;
    
    // CPU contribution (40% of compute score)
    if (!cpu_history_.empty()) {
        float avg_cpu_usage = 0.0f;
        for (const auto& entry : cpu_history_) {
            avg_cpu_usage += entry.second.usage_percent;
        }
        avg_cpu_usage /= cpu_history_.size();
        
        // Higher usage = lower score (more loaded)
        float cpu_score = std::max(0.0f, (100.0f - avg_cpu_usage) / 100.0f);
        score += cpu_score * 0.4f;
    }
    
    // GPU contribution (60% of compute score)
    if (!gpu_history_.empty()) {
        float avg_gpu_usage = 0.0f;
        for (const auto& entry : gpu_history_) {
            avg_gpu_usage += entry.second.usage_percent;
        }
        avg_gpu_usage /= gpu_history_.size();
        
        // Higher usage = lower score (more loaded)
        float gpu_score = std::max(0.0f, (100.0f - avg_gpu_usage) / 100.0f);
        score += gpu_score * 0.6f;
    }
    
    return std::min(1.0f, std::max(0.0f, score));
}

float PerformanceProfiler::calculate_memory_score() const {
    float score = 0.0f;
    
    if (!memory_history_.empty()) {
        float avg_memory_usage = 0.0f;
        for (const auto& entry : memory_history_) {
            avg_memory_usage += entry.second.usage_percent;
        }
        avg_memory_usage /= memory_history_.size();
        
        // Lower usage = higher score (more available memory)
        score = std::max(0.0f, (100.0f - avg_memory_usage) / 100.0f);
    }
    
    return std::min(1.0f, std::max(0.0f, score));
}

float PerformanceProfiler::calculate_network_score() const {
    float score = 1.0f;  // Default to good network
    
    if (!network_metrics_history_.empty()) {
        float avg_latency = 0.0f;
        for (const auto& entry : network_metrics_history_) {
            avg_latency += entry.second.latency_ms;
        }
        avg_latency /= network_metrics_history_.size();
        
        // Lower latency = higher score
        score = std::max(0.0f, std::min(1.0f, (200.0f - avg_latency) / 200.0f));
    }
    
    return score;
}

void PerformanceProfiler::update_training_statistics(const TrainingStepMetrics& metrics) {
    // Update running averages and statistics
    training_stats_.total_steps++;
    training_stats_.total_samples += metrics.batch_size;
    training_stats_.total_training_time_ms += metrics.step_time_ms;
    
    if (training_stats_.total_steps > 0) {
        training_stats_.average_step_time_ms = 
            training_stats_.total_training_time_ms / training_stats_.total_steps;
        training_stats_.average_throughput_samples_per_sec = 
            static_cast<float>(training_stats_.total_samples * 1000) / training_stats_.total_training_time_ms;
    }
}

void PerformanceProfiler::update_network_statistics(const NetworkOperationMetrics& metrics) {
    // Update network operation statistics
    network_stats_.total_operations++;
    network_stats_.total_bytes_transferred += metrics.bytes_transferred;
    network_stats_.total_operation_time_ms += metrics.operation_time_ms;
    
    if (network_stats_.total_operations > 0) {
        network_stats_.average_operation_time_ms = 
            network_stats_.total_operation_time_ms / network_stats_.total_operations;
        network_stats_.average_bandwidth_mbps = 
            (static_cast<float>(network_stats_.total_bytes_transferred) * 8.0f / 1024.0f / 1024.0f) / 
            (network_stats_.total_operation_time_ms / 1000.0f);
    }
}

void PerformanceProfiler::reset_counters() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    training_stats_ = TrainingStatistics{};
    network_stats_ = NetworkStatistics{};
    
    cpu_history_.clear();
    gpu_history_.clear();
    memory_history_.clear();
    network_metrics_history_.clear();
    training_history_.clear();
    network_history_.clear();
}

void PerformanceProfiler::export_metrics(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for metrics export: " << filename << std::endl;
        return;
    }
    
    file << "Performance Metrics Export\n";
    file << "========================\n\n";
    
    // Export system capabilities
    auto caps = get_system_capabilities();
    file << "System Capabilities:\n";
    file << "- CPU Cores: " << caps.cpu_cores << "\n";
    file << "- CPU Frequency: " << caps.cpu_frequency_mhz << " MHz\n";
    file << "- Total Memory: " << caps.total_memory_gb << " GB\n";
    file << "- GPU Count: " << caps.gpu_count << "\n";
    if (caps.gpu_count > 0) {
        file << "- GPU Memory: " << caps.gpu_memory_gb << " GB\n";
        file << "- GPU Compute: " << caps.gpu_compute_capability << "\n";
    }
    file << "\n";
    
    // Export training statistics
    file << "Training Statistics:\n";
    file << "- Total Steps: " << training_stats_.total_steps << "\n";
    file << "- Total Samples: " << training_stats_.total_samples << "\n";
    file << "- Average Step Time: " << training_stats_.average_step_time_ms << " ms\n";
    file << "- Average Throughput: " << training_stats_.average_throughput_samples_per_sec << " samples/sec\n";
    file << "\n";
    
    // Export network statistics
    file << "Network Statistics:\n";
    file << "- Total Operations: " << network_stats_.total_operations << "\n";
    file << "- Total Bytes: " << network_stats_.total_bytes_transferred << "\n";
    file << "- Average Operation Time: " << network_stats_.average_operation_time_ms << " ms\n";
    file << "- Average Bandwidth: " << network_stats_.average_bandwidth_mbps << " Mbps\n";
    
    file.close();
    std::cout << "Metrics exported to: " << filename << std::endl;
}

} // namespace profiling
