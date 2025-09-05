#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <functional>
#include <future>
#include "types.hpp"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace async_data {

struct BatchData {
    std::vector<std::vector<float>> input_data;
    std::vector<std::vector<float>> target_data;
    std::vector<size_t> sequence_lengths;
    size_t batch_size = 0;
    size_t max_sequence_length = 0;
    bool is_valid = false;
    
    // GPU memory pointers for zero-copy transfers
    #ifdef CUDA_AVAILABLE
    void* gpu_input_ptr = nullptr;
    void* gpu_target_ptr = nullptr;
    cudaStream_t cuda_stream = nullptr;
    #endif
};

struct DataLoaderConfig {
    size_t num_worker_threads = 4;
    size_t prefetch_queue_size = 8;
    size_t batch_size = 32;
    size_t max_sequence_length = 512;
    bool enable_gpu_prefetch = true;
    bool enable_data_augmentation = false;
    bool shuffle_data = true;
    float data_augmentation_probability = 0.1f;
    
    // Memory management
    bool use_pinned_memory = true;
    size_t memory_pool_size_mb = 512;
};

struct LoaderStats {
    std::atomic<size_t> batches_loaded{0};
    std::atomic<size_t> batches_served{0};
    std::atomic<size_t> cache_hits{0};
    std::atomic<size_t> cache_misses{0};
    std::atomic<float> average_load_time_ms{0.0f};
    std::atomic<float> queue_utilization{0.0f};
    std::atomic<bool> is_running{false};
};

// Data source interface for flexibility
class DataSource {
public:
    virtual ~DataSource() = default;
    virtual bool has_next_batch() = 0;
    virtual BatchData load_next_batch() = 0;
    virtual void reset() = 0;
    virtual size_t get_total_samples() const = 0;
    virtual size_t get_current_epoch() const = 0;
};

// File-based data source implementation
class FileDataSource : public DataSource {
private:
    std::string data_file_path_;
    std::vector<std::string> data_lines_;
    size_t current_index_;
    size_t current_epoch_;
    DataLoaderConfig config_;
    
public:
    explicit FileDataSource(const std::string& file_path, const DataLoaderConfig& config);
    
    bool has_next_batch() override;
    BatchData load_next_batch() override;
    void reset() override;
    size_t get_total_samples() const override;
    size_t get_current_epoch() const override;
    
private:
    void load_data_file();
    void shuffle_data();
    BatchData create_batch_from_lines(const std::vector<std::string>& lines);
};

class AsyncDataLoader {
private:
    DataLoaderConfig config_;
    std::unique_ptr<DataSource> data_source_;
    
    // Threading components
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> should_stop_;
    std::atomic<bool> is_running_;
    
    // Prefetch queue
    std::queue<std::future<BatchData>> prefetch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_not_empty_;
    std::condition_variable queue_not_full_;
    
    // GPU memory management
    #ifdef CUDA_AVAILABLE
    std::vector<cudaStream_t> cuda_streams_;
    void* gpu_memory_pool_ = nullptr;
    size_t gpu_memory_offset_ = 0;
    std::mutex gpu_memory_mutex_;
    #endif
    
    // Statistics and monitoring
    LoaderStats stats_;
    std::chrono::steady_clock::time_point last_stats_update_;
    
    // Memory pool for CPU data
    std::vector<std::unique_ptr<float[]>> cpu_memory_pool_;
    std::queue<size_t> available_memory_slots_;
    std::mutex memory_pool_mutex_;
    
public:
    explicit AsyncDataLoader(std::unique_ptr<DataSource> data_source, const DataLoaderConfig& config);
    ~AsyncDataLoader();
    
    /**
     * Start the prefetch pipeline with background threads
     * @return True if started successfully
     */
    bool start_prefetch_pipeline();
    
    /**
     * Stop the prefetch pipeline and cleanup resources
     */
    void stop_prefetch_pipeline();
    
    /**
     * Get the next batch (blocks if no batch is ready)
     * @param timeout_ms Maximum time to wait for a batch (0 = no timeout)
     * @return BatchData or invalid batch if timeout/error
     */
    BatchData get_next_batch(size_t timeout_ms = 0);
    
    /**
     * Try to get the next batch without blocking
     * @return BatchData or invalid batch if none available
     */
    BatchData try_get_next_batch();
    
    /**
     * Check if more batches are available
     * @return True if more batches can be loaded
     */
    bool has_more_batches() const;
    
    /**
     * Reset data source to beginning
     */
    void reset_data_source();
    
    /**
     * Get current loader statistics
     * @return LoaderStats with current metrics
     */
    LoaderStats get_statistics() const;
    
    /**
     * Update loader configuration (requires restart)
     * @param config New configuration
     */
    void update_config(const DataLoaderConfig& config);
    
    /**
     * Get current configuration
     * @return Current loader configuration
     */
    const DataLoaderConfig& get_config() const { return config_; }
    
    /**
     * Prefetch specific number of batches
     * @param num_batches Number of batches to prefetch
     */
    void prefetch_batches(size_t num_batches);
    
    /**
     * Get queue utilization (0.0 - 1.0)
     * @return Current queue fill ratio
     */
    float get_queue_utilization() const;

private:
    /**
     * Worker thread function for loading batches
     */
    void worker_thread_function();
    
    /**
     * Load a single batch asynchronously
     * @return Future containing the loaded batch
     */
    std::future<BatchData> load_batch_async();
    
    /**
     * Apply data augmentation to batch
     * @param batch Batch to augment
     */
    void apply_data_augmentation(BatchData& batch);
    
    /**
     * Transfer batch to GPU memory
     * @param batch Batch to transfer
     * @return True if transfer successful
     */
    bool transfer_batch_to_gpu(BatchData& batch);
    
    /**
     * Allocate GPU memory for batch
     * @param batch Batch requiring GPU memory
     * @return True if allocation successful
     */
    bool allocate_gpu_memory_for_batch(BatchData& batch);
    
    /**
     * Free GPU memory for batch
     * @param batch Batch to free GPU memory for
     */
    void free_gpu_memory_for_batch(BatchData& batch);
    
    /**
     * Initialize GPU resources
     * @return True if initialization successful
     */
    bool initialize_gpu_resources();
    
    /**
     * Cleanup GPU resources
     */
    void cleanup_gpu_resources();
    
    /**
     * Initialize CPU memory pool
     */
    void initialize_cpu_memory_pool();
    
    /**
     * Cleanup CPU memory pool
     */
    void cleanup_cpu_memory_pool();
    
    /**
     * Get memory slot from pool
     * @return Index of available memory slot, or SIZE_MAX if none available
     */
    size_t get_memory_slot();
    
    /**
     * Return memory slot to pool
     * @param slot_index Index of memory slot to return
     */
    void return_memory_slot(size_t slot_index);
    
    /**
     * Update performance statistics
     * @param load_time_ms Time taken to load batch
     */
    void update_statistics(float load_time_ms);
    
    /**
     * Check if queue has space for more batches
     * @return True if queue is not full
     */
    bool has_queue_space() const;
    
    /**
     * Wait for queue space to become available
     * @param timeout_ms Maximum time to wait
     * @return True if space became available
     */
    bool wait_for_queue_space(size_t timeout_ms);
};

/**
 * RAII wrapper for automatic batch cleanup
 */
class ScopedBatch {
private:
    BatchData batch_;
    AsyncDataLoader* loader_;
    
public:
    ScopedBatch(BatchData&& batch, AsyncDataLoader* loader)
        : batch_(std::move(batch)), loader_(loader) {}
    
    ~ScopedBatch() {
        // Automatic cleanup of GPU resources if needed
        #ifdef CUDA_AVAILABLE
        if (batch_.gpu_input_ptr) {
            cudaFree(batch_.gpu_input_ptr);
        }
        if (batch_.gpu_target_ptr) {
            cudaFree(batch_.gpu_target_ptr);
        }
        if (batch_.cuda_stream) {
            cudaStreamDestroy(batch_.cuda_stream);
        }
        #endif
    }
    
    const BatchData& get() const { return batch_; }
    BatchData& get() { return batch_; }
    
    bool is_valid() const { return batch_.is_valid; }
};

} // namespace async_data
