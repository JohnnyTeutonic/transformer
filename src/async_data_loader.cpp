#include "async_data_loader.hpp"
#include "logger.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>

namespace async_data {

// FileDataSource implementation
FileDataSource::FileDataSource(const std::string& file_path, const DataLoaderConfig& config)
    : data_file_path_(file_path), current_index_(0), current_epoch_(0), config_(config) {
    load_data_file();
    if (config_.shuffle_data) {
        shuffle_data();
    }
}

bool FileDataSource::has_next_batch() {
    return current_index_ < data_lines_.size();
}

BatchData FileDataSource::load_next_batch() {
    if (!has_next_batch()) {
        // End of epoch, reset and increment epoch counter
        reset();
        current_epoch_++;
        if (config_.shuffle_data) {
            shuffle_data();
        }
    }
    
    size_t batch_end = std::min(current_index_ + config_.batch_size, data_lines_.size());
    std::vector<std::string> batch_lines(data_lines_.begin() + current_index_, 
                                       data_lines_.begin() + batch_end);
    
    current_index_ = batch_end;
    
    return create_batch_from_lines(batch_lines);
}

void FileDataSource::reset() {
    current_index_ = 0;
}

size_t FileDataSource::get_total_samples() const {
    return data_lines_.size();
}

size_t FileDataSource::get_current_epoch() const {
    return current_epoch_;
}

void FileDataSource::load_data_file() {
    std::ifstream file(data_file_path_);
    if (!file.is_open()) {
        logger::log_error("Failed to open data file: " + data_file_path_);
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            data_lines_.push_back(line);
        }
    }
    
    logger::log_info("Loaded " + std::to_string(data_lines_.size()) + " samples from " + data_file_path_);
}

void FileDataSource::shuffle_data() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(data_lines_.begin(), data_lines_.end(), gen);
}

BatchData FileDataSource::create_batch_from_lines(const std::vector<std::string>& lines) {
    BatchData batch;
    batch.batch_size = lines.size();
    batch.max_sequence_length = config_.max_sequence_length;
    batch.is_valid = true;
    
    batch.input_data.resize(batch.batch_size);
    batch.target_data.resize(batch.batch_size);
    batch.sequence_lengths.resize(batch.batch_size);
    
    for (size_t i = 0; i < lines.size(); ++i) {
        // Simple tokenization (in practice, you'd use a proper tokenizer)
        std::istringstream iss(lines[i]);
        std::string token;
        std::vector<float> tokens;
        
        while (iss >> token && tokens.size() < config_.max_sequence_length) {
            // Convert token to float (simplified - in practice use vocabulary mapping)
            tokens.push_back(static_cast<float>(std::hash<std::string>{}(token) % 10000));
        }
        
        batch.sequence_lengths[i] = tokens.size();
        
        // Pad sequences to max length
        tokens.resize(config_.max_sequence_length, 0.0f);
        batch.input_data[i] = tokens;
        
        // Create target data (shifted by one position for language modeling)
        batch.target_data[i] = tokens;
        if (batch.target_data[i].size() > 1) {
            std::rotate(batch.target_data[i].begin(), batch.target_data[i].begin() + 1, batch.target_data[i].end());
            batch.target_data[i].back() = 0.0f; // End token
        }
    }
    
    return batch;
}

// AsyncDataLoader implementation
AsyncDataLoader::AsyncDataLoader(std::unique_ptr<DataSource> data_source, const DataLoaderConfig& config)
    : config_(config)
    , data_source_(std::move(data_source))
    , should_stop_(false)
    , is_running_(false)
    , gpu_memory_offset_(0)
    , last_stats_update_(std::chrono::steady_clock::now()) {
    
    initialize_cpu_memory_pool();
    
    #ifdef CUDA_AVAILABLE
    if (config_.enable_gpu_prefetch) {
        initialize_gpu_resources();
    }
    #endif
    
    logger::log_info("Initialized AsyncDataLoader with " + std::to_string(config_.num_worker_threads) + 
                     " worker threads, queue size: " + std::to_string(config_.prefetch_queue_size));
}

AsyncDataLoader::~AsyncDataLoader() {
    stop_prefetch_pipeline();
    cleanup_cpu_memory_pool();
    
    #ifdef CUDA_AVAILABLE
    cleanup_gpu_resources();
    #endif
}

bool AsyncDataLoader::start_prefetch_pipeline() {
    if (is_running_.load()) {
        logger::log_warning("Prefetch pipeline is already running");
        return false;
    }
    
    should_stop_.store(false);
    is_running_.store(true);
    stats_.is_running.store(true);
    
    // Start worker threads
    worker_threads_.reserve(config_.num_worker_threads);
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&AsyncDataLoader::worker_thread_function, this);
    }
    
    logger::log_info("Started prefetch pipeline with " + std::to_string(config_.num_worker_threads) + " worker threads");
    return true;
}

void AsyncDataLoader::stop_prefetch_pipeline() {
    if (!is_running_.load()) {
        return;
    }
    
    should_stop_.store(true);
    is_running_.store(false);
    stats_.is_running.store(false);
    
    // Wake up all waiting threads
    queue_not_empty_.notify_all();
    queue_not_full_.notify_all();
    
    // Join worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // Clear prefetch queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!prefetch_queue_.empty()) {
            prefetch_queue_.pop();
        }
    }
    
    logger::log_info("Stopped prefetch pipeline");
}

BatchData AsyncDataLoader::get_next_batch(size_t timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for batch to be available
    if (timeout_ms > 0) {
        auto timeout = std::chrono::milliseconds(timeout_ms);
        if (!queue_not_empty_.wait_for(lock, timeout, [this] { return !prefetch_queue_.empty() || should_stop_.load(); })) {
            logger::log_warning("Timeout waiting for batch");
            return BatchData{}; // Invalid batch
        }
    } else {
        queue_not_empty_.wait(lock, [this] { return !prefetch_queue_.empty() || should_stop_.load(); });
    }
    
    if (should_stop_.load() && prefetch_queue_.empty()) {
        return BatchData{}; // Invalid batch
    }
    
    // Get batch from queue
    auto batch_future = std::move(prefetch_queue_.front());
    prefetch_queue_.pop();
    
    // Update queue utilization
    stats_.queue_utilization.store(static_cast<float>(prefetch_queue_.size()) / config_.prefetch_queue_size);
    
    lock.unlock();
    queue_not_full_.notify_one(); // Signal that queue has space
    
    // Wait for batch to be ready
    try {
        BatchData batch = batch_future.get();
        stats_.batches_served.fetch_add(1);
        return batch;
    } catch (const std::exception& e) {
        logger::log_error("Error getting batch: " + std::string(e.what()));
        return BatchData{}; // Invalid batch
    }
}

BatchData AsyncDataLoader::try_get_next_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    if (prefetch_queue_.empty()) {
        return BatchData{}; // Invalid batch
    }
    
    auto batch_future = std::move(prefetch_queue_.front());
    prefetch_queue_.pop();
    
    // Update queue utilization
    stats_.queue_utilization.store(static_cast<float>(prefetch_queue_.size()) / config_.prefetch_queue_size);
    
    lock.unlock();
    queue_not_full_.notify_one();
    
    // Check if batch is ready
    if (batch_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        try {
            BatchData batch = batch_future.get();
            stats_.batches_served.fetch_add(1);
            return batch;
        } catch (const std::exception& e) {
            logger::log_error("Error getting batch: " + std::string(e.what()));
        }
    }
    
    return BatchData{}; // Invalid batch
}

bool AsyncDataLoader::has_more_batches() const {
    return data_source_->has_next_batch() || !prefetch_queue_.empty();
}

void AsyncDataLoader::reset_data_source() {
    data_source_->reset();
    logger::log_info("Reset data source");
}

LoaderStats AsyncDataLoader::get_statistics() const {
    return stats_;
}

void AsyncDataLoader::update_config(const DataLoaderConfig& config) {
    bool was_running = is_running_.load();
    
    if (was_running) {
        stop_prefetch_pipeline();
    }
    
    config_ = config;
    
    if (was_running) {
        start_prefetch_pipeline();
    }
    
    logger::log_info("Updated data loader configuration");
}

void AsyncDataLoader::prefetch_batches(size_t num_batches) {
    for (size_t i = 0; i < num_batches && has_more_batches(); ++i) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for queue space
        queue_not_full_.wait(lock, [this] { 
            return prefetch_queue_.size() < config_.prefetch_queue_size || should_stop_.load(); 
        });
        
        if (should_stop_.load()) {
            break;
        }
        
        // Add batch to queue
        prefetch_queue_.push(load_batch_async());
        
        lock.unlock();
        queue_not_empty_.notify_one();
    }
}

float AsyncDataLoader::get_queue_utilization() const {
    return stats_.queue_utilization.load();
}

void AsyncDataLoader::worker_thread_function() {
    logger::log_debug("Worker thread started");
    
    while (!should_stop_.load()) {
        try {
            // Check if we should load more batches
            if (!has_more_batches()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for queue space
            if (!queue_not_full_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return prefetch_queue_.size() < config_.prefetch_queue_size || should_stop_.load();
            })) {
                continue; // Timeout, try again
            }
            
            if (should_stop_.load()) {
                break;
            }
            
            // Add batch to queue
            prefetch_queue_.push(load_batch_async());
            
            // Update queue utilization
            stats_.queue_utilization.store(static_cast<float>(prefetch_queue_.size()) / config_.prefetch_queue_size);
            
            lock.unlock();
            queue_not_empty_.notify_one();
            
        } catch (const std::exception& e) {
            logger::log_error("Worker thread error: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    logger::log_debug("Worker thread stopped");
}

std::future<BatchData> AsyncDataLoader::load_batch_async() {
    return std::async(std::launch::async, [this]() {
        auto start_time = std::chrono::steady_clock::now();
        
        BatchData batch = data_source_->load_next_batch();
        
        if (batch.is_valid) {
            // Apply data augmentation if enabled
            if (config_.enable_data_augmentation) {
                apply_data_augmentation(batch);
            }
            
            // Transfer to GPU if enabled
            #ifdef CUDA_AVAILABLE
            if (config_.enable_gpu_prefetch) {
                transfer_batch_to_gpu(batch);
            }
            #endif
            
            stats_.batches_loaded.fetch_add(1);
        }
        
        // Update performance statistics
        auto end_time = std::chrono::steady_clock::now();
        float load_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        update_statistics(load_time_ms);
        
        return batch;
    });
}

void AsyncDataLoader::apply_data_augmentation(BatchData& batch) {
    if (!config_.enable_data_augmentation) {
        return;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < batch.batch_size; ++i) {
        if (prob_dist(gen) < config_.data_augmentation_probability) {
            // Simple augmentation: add small random noise
            std::normal_distribution<float> noise_dist(0.0f, 0.01f);
            
            for (float& value : batch.input_data[i]) {
                value += noise_dist(gen);
            }
        }
    }
}

bool AsyncDataLoader::transfer_batch_to_gpu(BatchData& batch) {
    #ifdef CUDA_AVAILABLE
    if (!allocate_gpu_memory_for_batch(batch)) {
        return false;
    }
    
    // Create CUDA stream for async transfer
    cudaStreamCreate(&batch.cuda_stream);
    
    // Calculate sizes
    size_t input_size = batch.batch_size * batch.max_sequence_length * sizeof(float);
    size_t target_size = batch.batch_size * batch.max_sequence_length * sizeof(float);
    
    // Flatten input data for GPU transfer
    std::vector<float> flat_input;
    std::vector<float> flat_target;
    flat_input.reserve(batch.batch_size * batch.max_sequence_length);
    flat_target.reserve(batch.batch_size * batch.max_sequence_length);
    
    for (const auto& input : batch.input_data) {
        flat_input.insert(flat_input.end(), input.begin(), input.end());
    }
    for (const auto& target : batch.target_data) {
        flat_target.insert(flat_target.end(), target.begin(), target.end());
    }
    
    // Async memory transfer
    cudaError_t result1 = cudaMemcpyAsync(batch.gpu_input_ptr, flat_input.data(), input_size, 
                                         cudaMemcpyHostToDevice, batch.cuda_stream);
    cudaError_t result2 = cudaMemcpyAsync(batch.gpu_target_ptr, flat_target.data(), target_size, 
                                         cudaMemcpyHostToDevice, batch.cuda_stream);
    
    if (result1 != cudaSuccess || result2 != cudaSuccess) {
        logger::log_error("Failed to transfer batch to GPU");
        free_gpu_memory_for_batch(batch);
        return false;
    }
    
    return true;
    #else
    return false;
    #endif
}

bool AsyncDataLoader::allocate_gpu_memory_for_batch(BatchData& batch) {
    #ifdef CUDA_AVAILABLE
    size_t input_size = batch.batch_size * batch.max_sequence_length * sizeof(float);
    size_t target_size = batch.batch_size * batch.max_sequence_length * sizeof(float);
    
    cudaError_t result1 = cudaMalloc(&batch.gpu_input_ptr, input_size);
    cudaError_t result2 = cudaMalloc(&batch.gpu_target_ptr, target_size);
    
    if (result1 != cudaSuccess || result2 != cudaSuccess) {
        logger::log_error("Failed to allocate GPU memory for batch");
        if (batch.gpu_input_ptr) cudaFree(batch.gpu_input_ptr);
        if (batch.gpu_target_ptr) cudaFree(batch.gpu_target_ptr);
        batch.gpu_input_ptr = nullptr;
        batch.gpu_target_ptr = nullptr;
        return false;
    }
    
    return true;
    #else
    return false;
    #endif
}

void AsyncDataLoader::free_gpu_memory_for_batch(BatchData& batch) {
    #ifdef CUDA_AVAILABLE
    if (batch.gpu_input_ptr) {
        cudaFree(batch.gpu_input_ptr);
        batch.gpu_input_ptr = nullptr;
    }
    if (batch.gpu_target_ptr) {
        cudaFree(batch.gpu_target_ptr);
        batch.gpu_target_ptr = nullptr;
    }
    if (batch.cuda_stream) {
        cudaStreamDestroy(batch.cuda_stream);
        batch.cuda_stream = nullptr;
    }
    #endif
}

bool AsyncDataLoader::initialize_gpu_resources() {
    #ifdef CUDA_AVAILABLE
    // Allocate GPU memory pool
    size_t pool_size = config_.memory_pool_size_mb * 1024 * 1024;
    cudaError_t result = cudaMalloc(&gpu_memory_pool_, pool_size);
    
    if (result != cudaSuccess) {
        logger::log_error("Failed to allocate GPU memory pool: " + std::string(cudaGetErrorString(result)));
        return false;
    }
    
    // Create CUDA streams for async operations
    cuda_streams_.resize(config_.num_worker_threads);
    for (auto& stream : cuda_streams_) {
        cudaStreamCreate(&stream);
    }
    
    logger::log_info("Initialized GPU resources with " + std::to_string(config_.memory_pool_size_mb) + "MB memory pool");
    return true;
    #else
    return false;
    #endif
}

void AsyncDataLoader::cleanup_gpu_resources() {
    #ifdef CUDA_AVAILABLE
    if (gpu_memory_pool_) {
        cudaFree(gpu_memory_pool_);
        gpu_memory_pool_ = nullptr;
    }
    
    for (auto& stream : cuda_streams_) {
        cudaStreamDestroy(stream);
    }
    cuda_streams_.clear();
    #endif
}

void AsyncDataLoader::initialize_cpu_memory_pool() {
    size_t num_slots = config_.prefetch_queue_size * 2; // Extra slots for safety
    cpu_memory_pool_.resize(num_slots);
    
    size_t slot_size = config_.batch_size * config_.max_sequence_length * 2; // Input + target
    
    for (size_t i = 0; i < num_slots; ++i) {
        if (config_.use_pinned_memory) {
            #ifdef CUDA_AVAILABLE
            float* ptr;
            cudaError_t result = cudaMallocHost(reinterpret_cast<void**>(&ptr), slot_size * sizeof(float));
            if (result == cudaSuccess) {
                cpu_memory_pool_[i] = std::unique_ptr<float[]>(ptr);
            } else {
                cpu_memory_pool_[i] = std::make_unique<float[]>(slot_size);
            }
            #else
            cpu_memory_pool_[i] = std::make_unique<float[]>(slot_size);
            #endif
        } else {
            cpu_memory_pool_[i] = std::make_unique<float[]>(slot_size);
        }
        
        available_memory_slots_.push(i);
    }
    
    logger::log_info("Initialized CPU memory pool with " + std::to_string(num_slots) + " slots");
}

void AsyncDataLoader::cleanup_cpu_memory_pool() {
    if (config_.use_pinned_memory) {
        #ifdef CUDA_AVAILABLE
        for (auto& ptr : cpu_memory_pool_) {
            if (ptr) {
                cudaFreeHost(ptr.release());
            }
        }
        #endif
    }
    cpu_memory_pool_.clear();
    
    // Clear available slots queue
    std::queue<size_t> empty;
    available_memory_slots_.swap(empty);
}

size_t AsyncDataLoader::get_memory_slot() {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    
    if (available_memory_slots_.empty()) {
        return SIZE_MAX; // No slots available
    }
    
    size_t slot = available_memory_slots_.front();
    available_memory_slots_.pop();
    return slot;
}

void AsyncDataLoader::return_memory_slot(size_t slot_index) {
    std::lock_guard<std::mutex> lock(memory_pool_mutex_);
    available_memory_slots_.push(slot_index);
}

void AsyncDataLoader::update_statistics(float load_time_ms) {
    // Update average load time with exponential moving average
    float current_avg = stats_.average_load_time_ms.load();
    float new_avg = (current_avg * 0.9f) + (load_time_ms * 0.1f);
    stats_.average_load_time_ms.store(new_avg);
    
    // Update other statistics periodically
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_update_).count() >= 1) {
        // Update queue utilization
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stats_.queue_utilization.store(static_cast<float>(prefetch_queue_.size()) / config_.prefetch_queue_size);
        }
        
        last_stats_update_ = now;
    }
}

bool AsyncDataLoader::has_queue_space() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return prefetch_queue_.size() < config_.prefetch_queue_size;
}

bool AsyncDataLoader::wait_for_queue_space(size_t timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    if (timeout_ms > 0) {
        auto timeout = std::chrono::milliseconds(timeout_ms);
        return queue_not_full_.wait_for(lock, timeout, [this] { 
            return prefetch_queue_.size() < config_.prefetch_queue_size || should_stop_.load(); 
        });
    } else {
        queue_not_full_.wait(lock, [this] { 
            return prefetch_queue_.size() < config_.prefetch_queue_size || should_stop_.load(); 
        });
        return true;
    }
}

} // namespace async_data
