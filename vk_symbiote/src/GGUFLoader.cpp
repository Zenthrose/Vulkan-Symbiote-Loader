#include "GGUFLoader.h"
#include "ConfigManager.h"
#include "Utils.h"
#include "../../compression/include/Blosc2Compression.h"
#include "../../compression/include/ZFPCompression.h"
#include "../../compression/include/HybridCompression.h"
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <thread>
#include <future>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <list>
#include <chrono>
#include <iostream>
#include <shared_mutex>
#include <functional>
#include <optional>
#include <regex>

namespace vk_symbiote {

// ============================================================================
// GGUFMetadataKV Implementation
// ============================================================================
std::string GGUFMetadataKV::get_string() const {
    if (value_type != GGUFValueType::STRING || raw_value.size() < 8) return "";
    uint64_t len = *reinterpret_cast<const uint64_t*>(raw_value.data());
    if (len > raw_value.size() - 8) return "";
    return std::string(reinterpret_cast<const char*>(raw_value.data() + 8), len);
}

uint32_t GGUFMetadataKV::get_uint32() const {
    if (value_type != GGUFValueType::UINT32 || raw_value.size() < 4) return 0;
    return *reinterpret_cast<const uint32_t*>(raw_value.data());
}

int32_t GGUFMetadataKV::get_int32() const {
    if (value_type != GGUFValueType::INT32 || raw_value.size() < 4) return 0;
    return *reinterpret_cast<const int32_t*>(raw_value.data());
}

float GGUFMetadataKV::get_float32() const {
    if (value_type != GGUFValueType::FLOAT32 || raw_value.size() < 4) return 0.0f;
    return *reinterpret_cast<const float*>(raw_value.data());
}

bool GGUFMetadataKV::get_bool() const {
    if (value_type != GGUFValueType::BOOL || raw_value.size() < 1) return false;
    return raw_value[0] != 0;
}

std::vector<uint8_t> GGUFMetadataKV::get_array_data() const {
    if (value_type != GGUFValueType::ARRAY || raw_value.size() < 12) return {};
    return std::vector<uint8_t>(raw_value.begin() + 12, raw_value.end());
}

GGUFValueType GGUFMetadataKV::get_array_element_type() const {
    if (value_type != GGUFValueType::ARRAY || raw_value.size() < 4) 
        return GGUFValueType::UINT8;
    return static_cast<GGUFValueType>(*reinterpret_cast<const uint32_t*>(raw_value.data()));
}

// ============================================================================
// WorkStealingThreadPool Implementation
// ============================================================================
class WorkStealingThreadPool {
public:
    struct TaskQueue {
        std::queue<std::function<void()>> tasks;
        std::mutex mutex;
    };

    explicit WorkStealingThreadPool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_(false), active_tasks_(0), next_queue_(0) {
        for (size_t i = 0; i < num_threads; ++i) {
            queues_.push_back(std::make_unique<TaskQueue>());
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~WorkStealingThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> result = task->get_future();

        size_t queue_idx = next_queue_.fetch_add(1) % queues_.size();
        {
            std::lock_guard<std::mutex> lock(queues_[queue_idx]->mutex);
            if (stop_) throw std::runtime_error("Cannot enqueue on stopped WorkStealingThreadPool");
            queues_[queue_idx]->tasks.emplace([task]() { (*task)(); });
            ++active_tasks_;
        }
        cv_.notify_one();
        return result;
    }

    size_t active_tasks() const { return active_tasks_.load(); }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mutex_);
        wait_cv_.wait(lock, [this] { return active_tasks_ == 0; });
    }

    size_t num_threads() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<TaskQueue>> queues_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable wait_cv_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;
    std::atomic<size_t> next_queue_;

    void worker_loop(size_t worker_id) {
        while (true) {
            std::function<void()> task;
            bool found_task = false;

            // Try local queue first
            {
                std::lock_guard<std::mutex> lock(queues_[worker_id]->mutex);
                if (!queues_[worker_id]->tasks.empty()) {
                    task = std::move(queues_[worker_id]->tasks.front());
                    queues_[worker_id]->tasks.pop();
                    found_task = true;
                }
            }

            // Try stealing from other queues if local is empty
            if (!found_task) {
                for (size_t i = 0; i < queues_.size() && !found_task; ++i) {
                    if (i == worker_id) continue;
                    std::lock_guard<std::mutex> lock(queues_[i]->mutex);
                    if (!queues_[i]->tasks.empty()) {
                        task = std::move(queues_[i]->tasks.front());
                        queues_[i]->tasks.pop();
                        found_task = true;
                    }
                }
            }

            // Wait for global signal if no work found
            if (!found_task) {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this, worker_id] {
                    return stop_ || !queues_[worker_id]->tasks.empty();
                });
                if (stop_) return;
                continue;
            }

            task();
            --active_tasks_;
            wait_cv_.notify_all();
        }
    }
};

// ============================================================================
// TensorLRUCache Implementation
// ============================================================================
TensorLRUCache::TensorLRUCache(const TensorCacheConfig& config) 
    : config_(config), current_memory_(0), hits_(0), misses_(0), evictions_(0) {}

bool TensorLRUCache::get(const std::string& tensor_name, std::vector<float>& out_data) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = cache_map_.find(tensor_name);
    if (it == cache_map_.end()) {
        ++misses_;
        return false;
    }
    
    // Check TTL
    if (config_.ttl_seconds > 0) {
        uint64_t age_sec = (get_current_time_ns() - it->second->entry.timestamp) / 1000000000ULL;
        if (age_sec > config_.ttl_seconds) {
            // Expired - evict
            current_memory_ -= it->second->entry.memory_cost;
            cache_list_.erase(it->second);
            cache_map_.erase(it);
            ++misses_;
            return false;
        }
    }
    
    // Move to front (most recently used)
    cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
    out_data = it->second->entry.data;
    it->second->entry.timestamp = get_current_time_ns();
    it->second->entry.access_count++;
    ++hits_;
    return true;
}

void TensorLRUCache::put(const std::string& tensor_name, const std::vector<float>& data,
                         const GGUFTensorInfo& info) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    size_t memory_cost = data.size() * sizeof(float);
    
    auto it = cache_map_.find(tensor_name);
    if (it != cache_map_.end()) {
        // Update existing entry
        current_memory_ -= it->second->entry.memory_cost;
        it->second->entry.data = data;
        it->second->entry.memory_cost = memory_cost;
        it->second->entry.timestamp = get_current_time_ns();
        it->second->entry.tensor_info = info;
        current_memory_ += memory_cost;
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return;
    }
    
    // Evict oldest if at capacity
    while (should_evict(memory_cost)) {
        if (cache_list_.empty()) break;
        evict_oldest();
    }
    
    // Insert new entry
    CacheNode node;
    node.key = tensor_name;
    node.entry.data = data;
    node.entry.memory_cost = memory_cost;
    node.entry.timestamp = get_current_time_ns();
    node.entry.access_count = 0;
    node.entry.tensor_info = info;
    
    cache_list_.push_front(std::move(node));
    cache_map_[tensor_name] = cache_list_.begin();
    current_memory_ += memory_cost;
}

bool TensorLRUCache::contains(const std::string& tensor_name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cache_map_.find(tensor_name) != cache_map_.end();
}

void TensorLRUCache::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cache_map_.clear();
    cache_list_.clear();
    current_memory_ = 0;
}

TensorLRUCache::Stats TensorLRUCache::get_stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    Stats stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.evictions = evictions_;
    stats.current_entries = cache_list_.size();
    stats.current_memory_bytes = current_memory_;
    return stats;
}

void TensorLRUCache::touch(const std::string& tensor_name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = cache_map_.find(tensor_name);
    if (it != cache_map_.end()) {
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        it->second->entry.timestamp = get_current_time_ns();
    }
}

size_t TensorLRUCache::remove_expired() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (config_.ttl_seconds == 0) return 0;
    
    size_t removed = 0;
    uint64_t now = get_current_time_ns();
    auto it = cache_list_.begin();
    while (it != cache_list_.end()) {
        uint64_t age_sec = (now - it->entry.timestamp) / 1000000000ULL;
        if (age_sec > config_.ttl_seconds) {
            current_memory_ -= it->entry.memory_cost;
            cache_map_.erase(it->key);
            it = cache_list_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

void TensorLRUCache::evict_oldest() {
    if (cache_list_.empty()) return;
    auto oldest = std::prev(cache_list_.end());
    current_memory_ -= oldest->entry.memory_cost;
    cache_map_.erase(oldest->key);
    cache_list_.erase(oldest);
    ++evictions_;
}

bool TensorLRUCache::should_evict(size_t new_memory) const {
    if (config_.max_entries > 0 && cache_list_.size() >= config_.max_entries) return true;
    if (config_.max_memory_bytes > 0 && current_memory_ + new_memory > config_.max_memory_bytes) return true;
    return false;
}

// ============================================================================
// StreamingTensorReader Implementation
// ============================================================================
StreamingTensorReader::StreamingTensorReader(const Path& file_path,
                                              WorkStealingThreadPool* thread_pool)
    : file_path_(file_path), thread_pool_(thread_pool) {
    file_.open(file_path_.string(), std::ios::binary | std::ios::in);
}

StreamingTensorReader::~StreamingTensorReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

Expected<std::vector<float>> StreamingTensorReader::read_tensor(const GGUFTensorInfo& tensor_info) {
    std::vector<uint8_t> raw_data(tensor_info.data_size);
    {
        std::lock_guard<std::mutex> lock(file_mutex_);
        file_.seekg(tensor_info.data_offset);
        file_.read(reinterpret_cast<char*>(raw_data.data()), tensor_info.data_size);
        if (!file_) {
            return -1;  // Error code
        }
    }
    
    // Decompress if needed
    if (tensor_info.is_compressed) {
        return decompress_data(raw_data, tensor_info);
    }
    
    // Dequantize if needed
    if (tensor_info.data_type >= GGUFValueType::Q4_0 && 
        tensor_info.data_type <= GGUFValueType::IQ4_XS_3) {
        return dequantize_data(raw_data, tensor_info);
    }
    
    // Raw float data
    std::vector<float> result;
    result.resize(raw_data.size() / sizeof(float));
    std::memcpy(result.data(), raw_data.data(), raw_data.size());
    
    return result;
}

std::vector<Expected<std::vector<float>>> StreamingTensorReader::read_tensors_parallel(
    const std::vector<GGUFTensorInfo>& tensors) {
    
    std::vector<Expected<std::vector<float>>> results(tensors.size());
    
    if (!thread_pool_) {
        // Fallback to sequential
        for (size_t i = 0; i < tensors.size(); ++i) {
            results[i] = read_tensor(tensors[i]);
        }
        return results;
    }
    
    // Submit all tasks
    std::vector<std::future<void>> futures;
    futures.reserve(tensors.size());
    
    for (size_t i = 0; i < tensors.size(); ++i) {
        futures.push_back(thread_pool_->enqueue([this, &tensors, &results, i]() {
            results[i] = read_tensor(tensors[i]);
        }));
    }
    
    // Wait for all to complete
    for (auto& f : futures) {
        f.wait();
    }
    
    return results;
}

Expected<std::vector<float>> StreamingTensorReader::read_tensor_hybrid(
    const GGUFTensorInfo& tensor_info) {
    
    if (!tensor_info.is_compressed) {
        return read_tensor(tensor_info);
    }
    
    // Read compressed data
    std::vector<uint8_t> compressed_data(tensor_info.data_size);
    {
        std::lock_guard<std::mutex> lock(file_mutex_);
        file_.seekg(tensor_info.data_offset);
        file_.read(reinterpret_cast<char*>(compressed_data.data()), tensor_info.data_size);
        if (!file_) {
            return -1;
        }
    }
    
    return decompress_data(compressed_data, tensor_info);
}

void StreamingTensorReader::read_tensor_async(const ReadRequest& request,
                                               std::function<void(ReadResult)> callback) {
    auto task = [this, request, callback]() {
        auto start = std::chrono::high_resolution_clock::now();
        ReadResult result;
        result.tensor_name = request.tensor_name;
        
        // Build temporary tensor info
        GGUFTensorInfo info;
        info.name = request.tensor_name;
        info.data_offset = request.file_offset;
        info.data_size = request.size;
        info.data_type = request.data_type;
        info.dimensions = request.dimensions;
        info.is_compressed = request.decompress;
        
        auto read_result = read_tensor(info);
        if (read_result.has_value()) {
            result.data = std::move(read_result.value());
            result.success = true;
        } else {
            result.success = false;
            result.error_msg = "Read failed";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.read_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if (callback) callback(result);
    };
    
    if (thread_pool_) {
        thread_pool_->enqueue(task);
    } else {
        std::thread(task).detach();
    }
}

void StreamingTensorReader::read_tensors_async_batch(
    const std::vector<ReadRequest>& requests,
    std::function<void(std::vector<ReadResult>)> batch_callback) {
    
    std::vector<ReadResult> results(requests.size());
    std::atomic<size_t> completed(0);
    
    for (size_t i = 0; i < requests.size(); ++i) {
        read_tensor_async(requests[i], [&, i](ReadResult result) {
            results[i] = std::move(result);
            if (++completed == requests.size()) {
                if (batch_callback) batch_callback(results);
            }
        });
    }
}

Expected<std::vector<float>> StreamingTensorReader::decompress_data(
    const std::vector<uint8_t>& compressed_data, const GGUFTensorInfo& info) {
    
    switch (info.compression_codec) {
        case 1: {  // Blosc2
            compression::Blosc2Compression blosc;
            size_t decompressed_size = info.dimensions.empty() ? compressed_data.size() : 
                info.dimensions[0] * sizeof(float);
            std::vector<float> float_data(decompressed_size / sizeof(float));
            if (!blosc.decompress(compressed_data.data(), compressed_data.size(), 
                                  decompressed_size, float_data.data())) {
                return -2;
            }
            return float_data;
        }
        case 2: {  // ZFP
            // Determine dimensionality from tensor info
            uint32_t dims = static_cast<uint32_t>(info.dimensions.size());
            if (dims == 0) return -3;
            
            size_t total_elements = 1;
            for (auto d : info.dimensions) total_elements *= d;
            
            compression::ZFPParams params = compression::ZFPParams::rate_mode(16.0);
            
            compression::ZFPCompression zfp(params);
            std::vector<float> output(total_elements);
            
            if (dims == 1) {
                if (!zfp.decompress_1d(compressed_data.data(), compressed_data.size(), 
                                       output.data(), info.dimensions[0])) {
                    return -4;
                }
            } else if (dims == 2) {
                if (!zfp.decompress_2d(compressed_data.data(), compressed_data.size(), 
                                       output.data(), info.dimensions[0], info.dimensions[1])) {
                    return -4;
                }
            } else if (dims == 3) {
                if (!zfp.decompress_3d(compressed_data.data(), compressed_data.size(), 
                                       output.data(), info.dimensions[0], info.dimensions[1], info.dimensions[2])) {
                    return -4;
                }
            } else {
                return -5;  // Unsupported dimensionality
            }
            
            return output;
        }
        case 3: {  // Hybrid
            compression::HybridCompression hybrid;
            size_t total_elements = 1;
            for (auto d : info.dimensions) total_elements *= d;
            std::vector<float> output(total_elements);
            if (!hybrid.decompress(compressed_data.data(), compressed_data.size(),
                                   total_elements * sizeof(float), output.data())) {
                return -6;
            }
            return output;
        }
        default:
            return -6;  // Unknown codec
    }
}

Expected<std::vector<float>> StreamingTensorReader::dequantize_data(
    const std::vector<uint8_t>& raw_data, const GGUFTensorInfo& info) {
    
    size_t num_elements = 1;
    for (auto d : info.dimensions) num_elements *= d;
    
    std::vector<float> result;
    result.reserve(num_elements);
    
    const uint8_t* ptr = raw_data.data();
    
    switch (info.data_type) {
        case GGUFValueType::Q4_0: {
            const size_t block_size = 32;
            const size_t num_blocks = (num_elements + block_size - 1) / block_size;
            
            for (size_t b = 0; b < num_blocks && (ptr - raw_data.data() + 2 + block_size/2) <= raw_data.size(); ++b) {
                float scale = *reinterpret_cast<const float*>(ptr);
                ptr += sizeof(float);
                
                for (size_t i = 0; i < block_size/2 && result.size() < num_elements; ++i) {
                    uint8_t packed = ptr[i];
                    int8_t q0 = (packed & 0x0F) - 8;
                    int8_t q1 = ((packed >> 4) & 0x0F) - 8;
                    result.push_back(q0 * scale);
                    if (result.size() < num_elements) result.push_back(q1 * scale);
                }
                ptr += block_size / 2;
            }
            break;
        }
        case GGUFValueType::Q4_1: {
            const size_t block_size = 32;
            const size_t num_blocks = (num_elements + block_size - 1) / block_size;
            
            for (size_t b = 0; b < num_blocks && (ptr - raw_data.data() + 4 + block_size/2) <= raw_data.size(); ++b) {
                float scale = *reinterpret_cast<const float*>(ptr);
                ptr += sizeof(float);
                float min = *reinterpret_cast<const float*>(ptr);
                ptr += sizeof(float);
                
                for (size_t i = 0; i < block_size/2 && result.size() < num_elements; ++i) {
                    uint8_t packed = ptr[i];
                    float q0 = static_cast<float>(packed & 0x0F);
                    float q1 = static_cast<float>((packed >> 4) & 0x0F);
                    result.push_back(q0 * scale + min);
                    if (result.size() < num_elements) result.push_back(q1 * scale + min);
                }
                ptr += block_size / 2;
            }
            break;
        }
        case GGUFValueType::Q8_0: {
            const size_t block_size = 32;
            const size_t num_blocks = (num_elements + block_size - 1) / block_size;
            
            for (size_t b = 0; b < num_blocks && (ptr - raw_data.data() + 2 + block_size) <= raw_data.size(); ++b) {
                float scale = *reinterpret_cast<const float*>(ptr);
                ptr += sizeof(float);
                
                for (size_t i = 0; i < block_size && result.size() < num_elements; ++i) {
                    int8_t q = static_cast<int8_t>(ptr[i]);
                    result.push_back(q * scale);
                }
                ptr += block_size;
            }
            break;
        }
        default:
            return -20;  // Unsupported quantization type
    }
    
    return result;
}

// ============================================================================
// GGUFLoaderImpl - Private Implementation
// ============================================================================
class GGUFLoaderImpl {
public:
    explicit GGUFLoaderImpl(const Path& file_path) : file_path_(file_path) {}
    ~GGUFLoaderImpl() { close(); }

    ExpectedVoid load() {
        file_.open(file_path_.string(), std::ios::binary);
        if (!file_.is_open()) {
            return -1;  // File open error
        }

        // Read header
        file_.read(reinterpret_cast<char*>(&header_), sizeof(GGUFHeader));
        if (header_.magic != 0x46554747) {  // "GGUF" in little-endian
            return -2;  // Invalid magic
        }

        version_ = header_.version;

        // Parse metadata
        if (!parse_metadata()) {
            return -3;  // Metadata parse error
        }

        // Parse tensor info
        if (!parse_tensor_info()) {
            return -4;  // Tensor info parse error
        }

        // Parse vocabulary from metadata
        parse_vocabulary();

        // Calculate data start offset
        data_start_offset_ = file_.tellg();

        // Update tensor data offsets
        for (auto& [name, info] : tensors_) {
            info.data_offset = data_start_offset_ + info.offset;
            info.data_size = calculate_tensor_data_size(info);
        }

        loaded_ = true;
        return ExpectedVoid();
    }

    ExpectedVoid close() {
        if (file_.is_open()) {
            file_.close();
        }
        loaded_ = false;
        return ExpectedVoid();
    }

    ModelConfig get_model_config() const {
        ModelConfig config;
        
        // Extract config from metadata
        if (auto kv = get_metadata_kv("llama.embedding_length")) {
            config.hidden_size = kv->get_uint32();
        } else if (auto kv2 = get_metadata_kv("general.embedding_length")) {
            config.hidden_size = kv2->get_uint32();
        }

        if (auto kv = get_metadata_kv("llama.block_count")) {
            config.num_layers = kv->get_uint32();
        } else if (auto kv2 = get_metadata_kv("general.block_count")) {
            config.num_layers = kv2->get_uint32();
        }

        if (auto kv = get_metadata_kv("llama.attention.head_count")) {
            config.num_attention_heads = kv->get_uint32();
        }

        if (auto kv = get_metadata_kv("llama.attention.head_count_kv")) {
            config.num_key_value_heads = kv->get_uint32();
        } else {
            config.num_key_value_heads = config.num_attention_heads;
        }

        if (auto kv = get_metadata_kv("llama.feed_forward_length")) {
            config.intermediate_size = kv->get_uint32();
        }

        if (auto kv = get_metadata_kv("llama.vocab_size")) {
            config.vocab_size = kv->get_uint32();
        } else if (auto kv2 = get_metadata_kv("tokenizer.ggml.tokens")) {
            // Count tokens in array
            auto arr_data = kv2->get_array_data();
            if (arr_data.size() >= 8) {
                config.vocab_size = *reinterpret_cast<const uint64_t*>(arr_data.data() + 4);
            }
        }

        if (auto kv = get_metadata_kv("llama.rope.freq_base")) {
            config.rope_theta = kv->get_float32();
        }

        if (auto kv = get_metadata_kv("llama.context_length")) {
            config.max_position_embeddings = kv->get_uint32();
        }

        if (auto kv = get_metadata_kv("general.architecture")) {
            config.model_type = kv->get_string();
        }

        config.head_dim = config.hidden_size / config.num_attention_heads;
        
        return config;
    }

    std::vector<PackMetadata> generate_packs(const ModelConfig& /*config*/) {
        std::vector<PackMetadata> packs;
        packs.reserve(tensors_.size());

        for (const auto& [name, info] : tensors_) {
            PackMetadata pack;
            pack.pack_id = std::hash<std::string>{}(name);
            pack.tensor_name = name;
            pack.layer_idx = extract_layer_index(name);
            pack.head_idx = extract_head_index(name);
            pack.compressed_size = info.data_size;
            pack.decompressed_size = info.data_size;
            pack.file_offset = info.data_offset;
            pack.base_priority = calculate_priority(name, info);
            pack.type = classify_tensor_type(name);
            pack.alignment = 256;

            packs.push_back(pack);
        }

        return packs;
    }

    const GGUFTensorInfo* get_tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    Expected<std::vector<float>> read_tensor_data(const GGUFTensorInfo& tensor, bool fp16_to_fp32) {
        if (!file_.is_open()) {
            return -10;
        }

        std::vector<uint8_t> raw_data(tensor.data_size);
        
        file_.seekg(tensor.data_offset);
        file_.read(reinterpret_cast<char*>(raw_data.data()), tensor.data_size);

        if (!file_) {
            return -11;
        }

        // Handle quantization
        if (tensor.data_type >= GGUFValueType::Q4_0 && tensor.data_type <= GGUFValueType::IQ4_XS_3) {
            return dequantize_tensor(raw_data, tensor);
        }

        // Handle FP16 to FP32 conversion
        if (tensor.data_type == GGUFValueType::FLOAT16 && fp16_to_fp32) {
            return convert_fp16_to_fp32(raw_data);
        }

        // Raw float data
        if (tensor.data_type == GGUFValueType::FLOAT32) {
            std::vector<float> result(raw_data.size() / sizeof(float));
            std::memcpy(result.data(), raw_data.data(), raw_data.size());
            return result;
        }

        return -12;  // Unsupported data type
    }

    const std::vector<VocabularyEntry>& get_vocabulary() const {
        return vocabulary_;
    }

    const TokenizerConfig& get_tokenizer_config() const {
        return tokenizer_config_;
    }

    uint64 tensor_count() const { return header_.tensor_count; }
    uint64 metadata_count() const { return header_.metadata_kv_count; }
    bool is_loaded() const { return loaded_; }
    uint32_t version() const { return version_; }

    std::vector<std::string> get_tensor_names() const {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (const auto& [name, _] : tensors_) {
            names.push_back(name);
        }
        return names;
    }

    std::vector<std::string> get_metadata_keys() const {
        std::vector<std::string> keys;
        keys.reserve(metadata_.size());
        for (const auto& [key, _] : metadata_) {
            keys.push_back(key);
        }
        return keys;
    }

    const GGUFMetadataKV* get_metadata(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it != metadata_.end()) {
            return &it->second;
        }
        return nullptr;
    }

private:
    Path file_path_;
    std::ifstream file_;
    GGUFHeader header_{};
    uint32_t version_ = 0;
    uint64_t data_start_offset_ = 0;
    bool loaded_ = false;

    std::unordered_map<std::string, GGUFMetadataKV> metadata_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;
    std::vector<VocabularyEntry> vocabulary_;
    TokenizerConfig tokenizer_config_;

    bool parse_metadata() {
        for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
            GGUFMetadataKV kv;
            
            // Read key length and key
            uint64_t key_len;
            file_.read(reinterpret_cast<char*>(&key_len), sizeof(uint64_t));
            if (!file_) return false;
            
            std::vector<char> key_data(key_len);
            file_.read(key_data.data(), key_len);
            if (!file_) return false;
            kv.key = std::string(key_data.data(), key_len);

            // Read value type
            uint32_t value_type_int;
            file_.read(reinterpret_cast<char*>(&value_type_int), sizeof(uint32_t));
            kv.value_type = static_cast<GGUFValueType>(value_type_int);

            // Read value
            if (!read_value(file_, kv.value_type, kv.raw_value)) {
                return false;
            }

            metadata_[kv.key] = std::move(kv);
        }
        return true;
    }

    bool parse_tensor_info() {
        for (uint64_t i = 0; i < header_.tensor_count; ++i) {
            GGUFTensorInfo info;

            // Read name
            uint64_t name_len;
            file_.read(reinterpret_cast<char*>(&name_len), sizeof(uint64_t));
            if (!file_) return false;
            
            std::vector<char> name_data(name_len);
            file_.read(name_data.data(), name_len);
            if (!file_) return false;
            info.name = std::string(name_data.data(), name_len);

            // Read dimensions
            uint32_t n_dims;
            file_.read(reinterpret_cast<char*>(&n_dims), sizeof(uint32_t));
            info.n_dimensions = n_dims;
            
            info.dimensions.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                file_.read(reinterpret_cast<char*>(&info.dimensions[d]), sizeof(uint64_t));
            }

            // Read data type
            uint32_t dtype;
            file_.read(reinterpret_cast<char*>(&dtype), sizeof(uint32_t));
            info.data_type = static_cast<GGUFValueType>(dtype);

            // Read offset
            file_.read(reinterpret_cast<char*>(&info.offset), sizeof(uint64_t));

            tensors_[info.name] = std::move(info);
        }
        return true;
    }

    bool read_value(std::ifstream& file, GGUFValueType type, std::vector<uint8_t>& out) {
        switch (type) {
            case GGUFValueType::UINT8:
            case GGUFValueType::INT8:
            case GGUFValueType::BOOL:
                out.resize(1);
                file.read(reinterpret_cast<char*>(out.data()), 1);
                return file.good();

            case GGUFValueType::UINT16:
            case GGUFValueType::INT16:
                out.resize(2);
                file.read(reinterpret_cast<char*>(out.data()), 2);
                return file.good();

            case GGUFValueType::UINT32:
            case GGUFValueType::INT32:
            case GGUFValueType::FLOAT32:
                out.resize(4);
                file.read(reinterpret_cast<char*>(out.data()), 4);
                return file.good();

            case GGUFValueType::UINT64:
            case GGUFValueType::INT64:
            case GGUFValueType::FLOAT64:
                out.resize(8);
                file.read(reinterpret_cast<char*>(out.data()), 8);
                return file.good();

            case GGUFValueType::STRING: {
                uint64_t len;
                file.read(reinterpret_cast<char*>(&len), sizeof(uint64_t));
                if (!file) return false;
                out.resize(8 + len);
                *reinterpret_cast<uint64_t*>(out.data()) = len;
                file.read(reinterpret_cast<char*>(out.data() + 8), len);
                return file.good();
            }

            case GGUFValueType::ARRAY: {
                uint32_t elem_type;
                uint64_t count;
                file.read(reinterpret_cast<char*>(&elem_type), sizeof(uint32_t));
                file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
                if (!file) return false;
                
                // Preallocate
                out.resize(12);
                *reinterpret_cast<uint32_t*>(out.data()) = elem_type;
                *reinterpret_cast<uint64_t*>(out.data() + 4) = count;
                
                // Read array elements
                GGUFValueType elem_type_enum = static_cast<GGUFValueType>(elem_type);
                for (uint64_t i = 0; i < count; ++i) {
                    std::vector<uint8_t> elem_data;
                    if (!read_value(file, elem_type_enum, elem_data)) {
                        return false;
                    }
                    out.insert(out.end(), elem_data.begin(), elem_data.end());
                }
                return true;
            }

            default:
                // For quantized types, treat as raw bytes
                return false;
        }
    }

    void parse_vocabulary() {
        // Find tokenizer.ggml.tokens
        auto tokens_kv = get_metadata_kv("tokenizer.ggml.tokens");
        if (!tokens_kv) {
            std::cerr << "[GGUFLoader] No vocabulary found in metadata" << std::endl;
            return;
        }

        // Parse tokenizer configuration
        if (auto kv = get_metadata_kv("tokenizer.ggml.model")) {
            tokenizer_config_.model_type = kv->get_string();
        }
        if (auto kv = get_metadata_kv("tokenizer.ggml.bos_token_id")) {
            tokenizer_config_.bos_token_id = kv->get_uint32();
        }
        if (auto kv = get_metadata_kv("tokenizer.ggml.eos_token_id")) {
            tokenizer_config_.eos_token_id = kv->get_uint32();
        }
        if (auto kv = get_metadata_kv("tokenizer.ggml.padding_token_id")) {
            tokenizer_config_.pad_token_id = kv->get_uint32();
        }
        if (auto kv = get_metadata_kv("tokenizer.ggml.unknown_token_id")) {
            tokenizer_config_.unk_token_id = kv->get_uint32();
        }
        if (auto kv = get_metadata_kv("tokenizer.ggml.add_bos_token")) {
            tokenizer_config_.add_bos_token = kv->get_bool();
        }
        if (auto kv = get_metadata_kv("tokenizer.ggml.add_eos_token")) {
            tokenizer_config_.add_eos_token = kv->get_bool();
        }
        if (auto kv = get_metadata_kv("tokenizer.chat_template")) {
            tokenizer_config_.chat_template = kv->get_string();
        }

        // Parse tokens array
        auto arr_data = tokens_kv->get_array_data();
        if (arr_data.size() < 8) {
            std::cerr << "[GGUFLoader] Invalid tokens array" << std::endl;
            return;
        }

        uint64_t count = *reinterpret_cast<const uint64_t*>(arr_data.data() + 4);
        vocabulary_.reserve(count);

        size_t offset = 8;
        for (uint64_t i = 0; i < count && offset < arr_data.size(); ++i) {
            if (offset + 8 > arr_data.size()) break;

            uint64_t str_len = *reinterpret_cast<const uint64_t*>(arr_data.data() + offset);
            offset += 8;

            if (offset + str_len > arr_data.size()) break;

            VocabularyEntry entry;
            entry.token.assign(reinterpret_cast<const char*>(arr_data.data() + offset), str_len);
            entry.bytes.assign(arr_data.data() + offset, arr_data.data() + offset + str_len);
            entry.score = 0.0f;
            entry.token_type = 0;
            entry.is_special = false;

            vocabulary_.push_back(std::move(entry));
            offset += str_len;
        }

        // Parse token scores if available
        if (auto scores_kv = get_metadata_kv("tokenizer.ggml.scores")) {
            auto scores_data = scores_kv->get_array_data();
            if (scores_data.size() >= 8) {
                uint64_t scores_count = *reinterpret_cast<const uint64_t*>(scores_data.data() + 4);
                size_t scores_offset = 8;
                for (size_t i = 0; i < scores_count && i < vocabulary_.size(); ++i) {
                    if (scores_offset + 4 > scores_data.size()) break;
                    vocabulary_[i].score = *reinterpret_cast<const float*>(scores_data.data() + scores_offset);
                    scores_offset += 4;
                }
            }
        }

        // Parse token types if available
        if (auto types_kv = get_metadata_kv("tokenizer.ggml.token_type")) {
            auto types_data = types_kv->get_array_data();
            if (types_data.size() >= 8) {
                uint64_t types_count = *reinterpret_cast<const uint64_t*>(types_data.data() + 4);
                size_t types_offset = 8;
                for (size_t i = 0; i < types_count && i < vocabulary_.size(); ++i) {
                    if (types_offset + 4 > types_data.size()) break;
                    vocabulary_[i].token_type = *reinterpret_cast<const uint32_t*>(types_data.data() + types_offset);
                    types_offset += 4;
                }
            }
        }

        std::cout << "[GGUFLoader] Parsed " << vocabulary_.size() << " vocabulary entries" << std::endl;
    }

    const GGUFMetadataKV* get_metadata_kv(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it != metadata_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    uint64_t calculate_tensor_data_size(const GGUFTensorInfo& info) const {
        size_t num_elements = 1;
        for (auto d : info.dimensions) num_elements *= d;

        switch (info.data_type) {
            case GGUFValueType::UINT8:
            case GGUFValueType::INT8:
                return num_elements;
            case GGUFValueType::UINT16:
            case GGUFValueType::INT16:
            case GGUFValueType::FLOAT16:
            case GGUFValueType::BFLOAT16:
                return num_elements * 2;
            case GGUFValueType::UINT32:
            case GGUFValueType::INT32:
            case GGUFValueType::FLOAT32:
                return num_elements * 4;
            case GGUFValueType::UINT64:
            case GGUFValueType::INT64:
            case GGUFValueType::FLOAT64:
                return num_elements * 8;
            case GGUFValueType::Q4_0:
                return (num_elements / 32) * (2 + 16) + ((num_elements % 32) ? (2 + 16) : 0);
            case GGUFValueType::Q4_1:
                return (num_elements / 32) * (2 + 2 + 16) + ((num_elements % 32) ? (2 + 2 + 16) : 0);
            case GGUFValueType::Q5_0:
                return (num_elements / 32) * (2 + 20) + ((num_elements % 32) ? (2 + 20) : 0);
            case GGUFValueType::Q5_1:
                return (num_elements / 32) * (2 + 2 + 20) + ((num_elements % 32) ? (2 + 2 + 20) : 0);
            case GGUFValueType::Q8_0:
                return (num_elements / 32) * (2 + 32) + ((num_elements % 32) ? (2 + 32) : 0);
            default:
                return num_elements * 4;  // Assume float32 for unknown
        }
    }

    Expected<std::vector<float>> dequantize_tensor(const std::vector<uint8_t>& data, 
                                                    const GGUFTensorInfo& info) {
        size_t num_elements = 1;
        for (auto d : info.dimensions) num_elements *= d;
        
        std::vector<float> result;
        result.reserve(num_elements);

        const uint8_t* ptr = data.data();

        // Helper: Convert FP16 to FP32
        auto fp16_to_fp32 = [](uint16_t half) -> float {
            uint32_t sign = (half >> 15) & 0x1;
            uint32_t exp = (half >> 10) & 0x1F;
            uint32_t mantissa = half & 0x3FF;
            
            if (exp == 0) {
                return mantissa == 0 ? 0.0f : std::ldexp((float)mantissa, -24);
            } else if (exp == 31) {
                return mantissa == 0 ? 
                    (sign ? -INFINITY : INFINITY) : 
                    (sign ? -NAN : NAN);
            }
            
            uint32_t float_bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mantissa << 13);
            float f;
            std::memcpy(&f, &float_bits, sizeof(float));
            return f;
        };

        switch (info.data_type) {
            case GGUFValueType::Q4_0: {
                const size_t block_size = 32;
                const size_t num_blocks = (num_elements + block_size - 1) / block_size;
                const size_t block_bytes = 2 + 16;  // scale (fp16) + quantized values
                
                for (size_t b = 0; b < num_blocks && (ptr - data.data() + block_bytes) <= data.size(); ++b) {
                    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(ptr);
                    float scale = fp16_to_fp32(scale_bits);
                    ptr += sizeof(uint16_t);
                    
                    for (size_t i = 0; i < block_size/2 && result.size() < num_elements; ++i) {
                        uint8_t packed = ptr[i];
                        int8_t q0 = (packed & 0x0F) - 8;
                        int8_t q1 = ((packed >> 4) & 0x0F) - 8;
                        result.push_back(q0 * scale);
                        if (result.size() < num_elements) result.push_back(q1 * scale);
                    }
                    ptr += block_size / 2;
                }
                break;
            }

            case GGUFValueType::Q4_1: {
                const size_t block_size = 32;
                const size_t num_blocks = (num_elements + block_size - 1) / block_size;
                const size_t block_bytes = 2 + 2 + 16;  // scale + min + quantized values (fp16)
                
                for (size_t b = 0; b < num_blocks && (ptr - data.data() + block_bytes) <= data.size(); ++b) {
                    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(ptr);
                    float scale = fp16_to_fp32(scale_bits);
                    ptr += sizeof(uint16_t);
                    uint16_t min_bits = *reinterpret_cast<const uint16_t*>(ptr);
                    float min = fp16_to_fp32(min_bits);
                    ptr += sizeof(uint16_t);
                    
                    for (size_t i = 0; i < block_size/2 && result.size() < num_elements; ++i) {
                        uint8_t packed = ptr[i];
                        float q0 = static_cast<float>(packed & 0x0F);
                        float q1 = static_cast<float>((packed >> 4) & 0x0F);
                        result.push_back(q0 * scale + min);
                        if (result.size() < num_elements) result.push_back(q1 * scale + min);
                    }
                    ptr += block_size / 2;
                }
                break;
            }

            case GGUFValueType::Q8_0: {
                const size_t block_size = 32;
                const size_t num_blocks = (num_elements + block_size - 1) / block_size;
                const size_t block_bytes = 2 + 32;  // scale (fp16) + quantized values
                
                for (size_t b = 0; b < num_blocks && (ptr - data.data() + block_bytes) <= data.size(); ++b) {
                    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(ptr);
                    float scale = fp16_to_fp32(scale_bits);
                    ptr += sizeof(uint16_t);
                    
                    for (size_t i = 0; i < block_size && result.size() < num_elements; ++i) {
                        int8_t q = static_cast<int8_t>(ptr[i]);
                        result.push_back(q * scale);
                    }
                    ptr += block_size;
                }
                break;
            }

            default:
                return -20;  // Unsupported quantization type
        }

        return result;
    }

    Expected<std::vector<float>> convert_fp16_to_fp32(const std::vector<uint8_t>& data) {
        size_t num_elements = data.size() / 2;
        std::vector<float> result;
        result.reserve(num_elements);

        for (size_t i = 0; i < num_elements; ++i) {
            uint16_t fp16 = *reinterpret_cast<const uint16_t*>(data.data() + i * 2);
            // Simple FP16 to FP32 conversion
            uint32_t sign = (fp16 >> 15) & 0x1;
            uint32_t exponent = (fp16 >> 10) & 0x1F;
            uint32_t mantissa = fp16 & 0x3FF;
            
            if (exponent == 0) {
                // Zero or subnormal
                if (mantissa == 0) {
                    result.push_back(sign ? -0.0f : 0.0f);
                } else {
                    // Subnormal number
                    float val = mantissa / 1024.0f * std::pow(2.0f, -14);
                    result.push_back(sign ? -val : val);
                }
            } else if (exponent == 31) {
                // Infinity or NaN
                if (mantissa == 0) {
                    result.push_back(sign ? -std::numeric_limits<float>::infinity() : 
                                            std::numeric_limits<float>::infinity());
                } else {
                    result.push_back(std::numeric_limits<float>::quiet_NaN());
                }
            } else {
                // Normal number
                float val = (1.0f + mantissa / 1024.0f) * std::pow(2.0f, exponent - 15);
                result.push_back(sign ? -val : val);
            }
        }

        return result;
    }

    uint32_t extract_layer_index(const std::string& name) const {
        // Extract layer number from names like "blk.0.attn_q.weight"
        std::regex layer_regex(R"(blk\.(\d+))");
        std::smatch match;
        if (std::regex_search(name, match, layer_regex)) {
            return std::stoul(match[1].str());
        }
        return 0;
    }

    uint32_t extract_head_index(const std::string& /*name*/) const {
        // Extract head index if present in tensor name
        return 0;  // Default to 0 for now
    }

    float calculate_priority(const std::string& name, const GGUFTensorInfo& /*info*/) const {
        float priority = 0.5f;
        
        // Higher priority for attention weights
        if (name.find("attn") != std::string::npos) {
            priority += 0.2f;
        }
        
        // Higher priority for embedding and output layers
        if (name.find("token_embd") != std::string::npos || 
            name.find("output") != std::string::npos) {
            priority += 0.3f;
        }
        
        // Adjust by layer depth (earlier layers slightly more important)
        uint32_t layer = extract_layer_index(name);
        if (layer < 4) {
            priority += 0.1f;
        }
        
        return std::min(1.0f, priority);
    }

    PackType classify_tensor_type(const std::string& name) const {
        if (name.find("attn_q") != std::string::npos) return PackType::ATTENTION_Q;
        if (name.find("attn_k") != std::string::npos) return PackType::ATTENTION_K;
        if (name.find("attn_v") != std::string::npos) return PackType::ATTENTION_V;
        if (name.find("attn_o") != std::string::npos || name.find("attn_output") != std::string::npos) 
            return PackType::ATTENTION_O;
        if (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos) 
            return PackType::FEED_FORWARD_GATE;
        if (name.find("ffn_down") != std::string::npos) return PackType::FEED_FORWARD_DOWN;
        if (name.find("ffn_norm") != std::string::npos || name.find("attn_norm") != std::string::npos)
            return PackType::NORM_GAMMA;
        if (name.find("token_embd") != std::string::npos) return PackType::EMBEDDING;
        if (name.find("output") != std::string::npos) return PackType::HEAD;
        return PackType::UNKNOWN;
    }
};

// ============================================================================
// GGUFLoader Public API Implementation
// ============================================================================
GGUFLoader::GGUFLoader(const Path& file_path)
    : file_path_(file_path),
      pimpl_(std::make_unique<GGUFLoaderImpl>(file_path)),
      tensor_cache_(std::make_unique<TensorLRUCache>()),
      thread_pool_(std::make_unique<WorkStealingThreadPool>()) {
    tensor_reader_ = std::make_unique<StreamingTensorReader>(file_path, thread_pool_.get());
}

GGUFLoader::~GGUFLoader() = default;

ExpectedVoid GGUFLoader::load() {
    auto result = pimpl_->load();
    if (result.has_value()) {
        loaded_ = true;
        version_ = pimpl_->version();
    }
    return result;
}

ExpectedVoid GGUFLoader::close() {
    loaded_ = false;
    return pimpl_->close();
}

ModelConfig GGUFLoader::get_model_config() const {
    return pimpl_->get_model_config();
}

std::vector<PackMetadata> GGUFLoader::generate_packs(const ModelConfig& config) {
    return pimpl_->generate_packs(config);
}

uint64 GGUFLoader::tensor_count() const {
    return pimpl_->tensor_count();
}

uint64 GGUFLoader::metadata_count() const {
    return pimpl_->metadata_count();
}

const GGUFTensorInfo* GGUFLoader::get_tensor(const std::string& name) const {
    return pimpl_->get_tensor(name);
}

Expected<std::vector<float>> GGUFLoader::read_tensor_data(const GGUFTensorInfo& tensor, 
                                                           bool fp16_to_fp32) {
    return pimpl_->read_tensor_data(tensor, fp16_to_fp32);
}

Expected<std::vector<float>> GGUFLoader::read_tensor_cached(const std::string& tensor_name,
                                                             bool fp16_to_fp32) {
    // Check cache first
    std::vector<float> cached_data;
    if (tensor_cache_->get(tensor_name, cached_data)) {
        return cached_data;
    }
    
    // Load from file
    const auto* tensor = get_tensor(tensor_name);
    if (!tensor) {
        return -30;  // Tensor not found
    }
    
    auto result = read_tensor_data(*tensor, fp16_to_fp32);
    if (result.has_value()) {
        // Add to cache
        tensor_cache_->put(tensor_name, result.value(), *tensor);
    }
    
    return result;
}

std::vector<Expected<std::vector<float>>> GGUFLoader::read_tensors_parallel(
    const std::vector<std::string>& tensor_names, bool fp16_to_fp32) {
    
    std::vector<Expected<std::vector<float>>> results(tensor_names.size());
    
    // First check cache for all tensors
    std::vector<std::string> uncached_names;
    std::vector<size_t> uncached_indices;
    
    for (size_t i = 0; i < tensor_names.size(); ++i) {
        std::vector<float> cached;
        if (tensor_cache_->get(tensor_names[i], cached)) {
            results[i] = std::move(cached);
        } else {
            uncached_names.push_back(tensor_names[i]);
            uncached_indices.push_back(i);
        }
    }
    
    // Load uncached tensors in parallel
    if (!uncached_names.empty()) {
        std::vector<std::future<void>> futures;
        futures.reserve(uncached_names.size());
        
        for (size_t i = 0; i < uncached_names.size(); ++i) {
            futures.push_back(thread_pool_->enqueue([this, &uncached_names, &results, 
                                                      &uncached_indices, i, fp16_to_fp32]() {
                const auto* tensor = get_tensor(uncached_names[i]);
                if (tensor) {
                    auto result = read_tensor_data(*tensor, fp16_to_fp32);
                    results[uncached_indices[i]] = std::move(result);
                    
                    // Cache successful loads
                    if (results[uncached_indices[i]].has_value()) {
                        tensor_cache_->put(uncached_names[i], 
                                          results[uncached_indices[i]].value(), 
                                          *tensor);
                    }
                } else {
                    results[uncached_indices[i]] = -30;  // Not found
                }
            }));
        }
        
        for (auto& f : futures) {
            f.wait();
        }
    }
    
    return results;
}

std::vector<Expected<std::vector<float>>> GGUFLoader::stream_tensors_parallel(
    const std::vector<std::string>& tensor_names,
    uint32_t /*num_chunks*/) {
    
    // Use StreamingTensorReader for async multi-thread chunk processing
    std::vector<StreamingTensorReader::ReadRequest> requests;
    requests.reserve(tensor_names.size());
    
    for (const auto& name : tensor_names) {
        const auto* tensor = get_tensor(name);
        if (tensor) {
            StreamingTensorReader::ReadRequest req;
            req.file_offset = tensor->data_offset;
            req.size = tensor->data_size;
            req.tensor_name = name;
            req.data_type = tensor->data_type;
            req.dimensions = tensor->dimensions;
            req.decompress = tensor->is_compressed;
            requests.push_back(std::move(req));
        }
    }
    
    std::vector<Expected<std::vector<float>>> results;
    results.reserve(tensor_names.size());
    
    std::mutex results_mutex;
    std::condition_variable cv;
    std::atomic<size_t> completed(0);
    
    tensor_reader_->read_tensors_async_batch(requests, 
        [&](std::vector<StreamingTensorReader::ReadResult> read_results) {
            std::lock_guard<std::mutex> lock(results_mutex);
            for (auto& rr : read_results) {
                if (rr.success) {
                    results.push_back(std::move(rr.data));
                } else {
                    results.push_back(-40);  // Read error
                }
            }
            completed.store(read_results.size());
            cv.notify_one();
        });
    
    // Wait for completion
    std::unique_lock<std::mutex> lock(results_mutex);
    cv.wait(lock, [&] { return completed.load() == requests.size(); });
    
    return results;
}

const std::vector<VocabularyEntry>& GGUFLoader::get_vocabulary() const {
    return pimpl_->get_vocabulary();
}

const TokenizerConfig& GGUFLoader::get_tokenizer_config() const {
    return pimpl_->get_tokenizer_config();
}

void GGUFLoader::clear_tensor_cache() {
    tensor_cache_->clear();
}

size_t GGUFLoader::get_tensor_cache_size() const {
    return tensor_cache_->get_stats().current_memory_bytes;
}

TensorLRUCache::Stats GGUFLoader::get_cache_stats() const {
    return tensor_cache_->get_stats();
}

void GGUFLoader::configure_cache(const TensorCacheConfig& config) {
    tensor_cache_ = std::make_unique<TensorLRUCache>(config);
}

ExpectedVoid GGUFLoader::add_shard(const Path& shard_path) {
    // Multi-shard support - for now just log
    std::cout << "[GGUFLoader] Shard support not fully implemented yet: " 
              << shard_path.string() << std::endl;
    return ExpectedVoid();
}

bool GGUFLoader::is_sharded() const {
    return false;  // Not yet implemented
}

std::vector<std::string> GGUFLoader::get_tensor_names() const {
    return pimpl_->get_tensor_names();
}

std::vector<std::string> GGUFLoader::get_metadata_keys() const {
    return pimpl_->get_metadata_keys();
}

const GGUFMetadataKV* GGUFLoader::get_metadata(const std::string& key) const {
    return pimpl_->get_metadata(key);
}

uint64 GGUFLoader::align_offset(uint64 offset, uint64 alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

uint32_t GGUFLoader::get_version() const {
    return version_;
}

bool GGUFLoader::is_valid() const {
    return loaded_;
}

} // namespace vk_symbiote
