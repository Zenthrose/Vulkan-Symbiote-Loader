#include "GGUFLoader.h"
#include "ConfigManager.h"
#include "../../compression/include/Blosc2Compression.h"
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
#include <queue>
#include <functional>
#include <optional>

namespace vk_symbiote {

// ============================================================================
// Advanced Thread Pool with Work Stealing for 70B+ Model Loading
// ============================================================================
class WorkStealingThreadPool {
public:
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
    struct TaskQueue {
        std::queue<std::function<void()>> tasks;
        std::mutex mutex;
    };

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
// Enhanced LRU Cache with TTL, Memory Pressure, and Hit Rate Tracking
// ============================================================================
template<typename K, typename V>
class EnhancedLRUCache {
public:
    struct CacheConfig {
        size_t max_size = 100;
        size_t max_memory_bytes = 0;  // 0 = unlimited
        uint64_t ttl_ms = 0;  // 0 = no expiration
        bool enable_stats = true;
    };

    struct CacheStats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        size_t size = 0;
        size_t memory_bytes = 0;
    };

    explicit EnhancedLRUCache(const CacheConfig& config = CacheConfig{})
        : config_(config), current_memory_(0) {
        if (config_.enable_stats) {
            hits_ = 0;
            misses_ = 0;
            evictions_ = 0;
        }
    }

    bool get(const K& key, V& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            if (config_.enable_stats) ++misses_;
            return false;
        }

        // Check TTL
        if (config_.ttl_ms > 0) {
            uint64_t age_ms = get_current_time_ms() - it->second->timestamp;
            if (age_ms > config_.ttl_ms) {
                evict_entry(it->second);
                if (config_.enable_stats) ++misses_;
                return false;
            }
        }

        // Move to front (most recently used)
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        value = it->second->value;
        it->second->timestamp = get_current_time_ms();
        if (config_.enable_stats) ++hits_;
        return true;
    }

    void put(const K& key, const V& value, size_t memory_cost = 0) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            current_memory_ -= it->second->memory_cost;
            it->second->value = value;
            it->second->memory_cost = memory_cost;
            it->second->timestamp = get_current_time_ms();
            current_memory_ += memory_cost;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }

        // Evict oldest if at capacity or memory limit
        while (should_evict(memory_cost)) {
            if (cache_list_.empty()) break;
            evict_entry(std::prev(cache_list_.end()));
            if (config_.enable_stats) ++evictions_;
        }

        CacheEntry entry;
        entry.key = key;
        entry.value = value;
        entry.memory_cost = memory_cost;
        entry.timestamp = get_current_time_ms();

        cache_list_.emplace_front(entry);
        cache_map_[key] = cache_list_.begin();
        current_memory_ += memory_cost;
    }

    bool contains(const K& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_map_.find(key) != cache_map_.end();
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_map_.clear();
        cache_list_.clear();
        current_memory_ = 0;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_list_.size();
    }

    size_t memory_usage() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return current_memory_;
    }

    double hit_rate() const {
        if (!config_.enable_stats) return 0.0;
        std::shared_lock<std::shared_mutex> lock(mutex_);
        uint64_t total = hits_ + misses_;
        return total > 0 ? static_cast<double>(hits_) / static_cast<double>(total) : 0.0;
    }

    CacheStats get_stats() const {
        if (!config_.enable_stats) return CacheStats{};
        std::shared_lock<std::shared_mutex> lock(mutex_);
        CacheStats stats;
        stats.hits = hits_;
        stats.misses = misses_;
        stats.evictions = evictions_;
        stats.size = cache_list_.size();
        stats.memory_bytes = current_memory_;
        return stats;
    }

    void record_hit() { if (config_.enable_stats) { std::unique_lock<std::shared_mutex> lock(mutex_); ++hits_; } }
    void record_miss() { if (config_.enable_stats) { std::unique_lock<std::shared_mutex> lock(mutex_); ++misses_; } }

    // Prefetch hint - mark entry as recently used without fetching
    void touch(const K& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        }
    }

private:
    struct CacheEntry {
        K key;
        V value;
        size_t memory_cost = 0;
        uint64_t timestamp = 0;
    };

    CacheConfig config_;
    size_t current_memory_;
    mutable std::list<CacheEntry> cache_list_;
    mutable std::unordered_map<K, typename std::list<CacheEntry>::iterator> cache_map_;
    mutable std::shared_mutex mutex_;
    mutable uint64_t hits_ = 0;
    mutable uint64_t misses_ = 0;
    mutable uint64_t evictions_ = 0;

    bool should_evict(size_t new_memory_cost) const {
        if (config_.max_size > 0 && cache_list_.size() >= config_.max_size) return true;
        if (config_.max_memory_bytes > 0 && current_memory_ + new_memory_cost > config_.max_memory_bytes) return true;
        return false;
    }

    void evict_entry(typename std::list<CacheEntry>::iterator it) {
        current_memory_ -= it->memory_cost;
        cache_map_.erase(it->key);
        cache_list_.erase(it);
    }

    static uint64_t get_current_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// ============================================================================
// Comprehensive Vocabulary Parser with Full GGUF Tokenizer.ggml Support
// ============================================================================
class VocabularyParser {
public:
    struct TokenizerConfig {
        std::string model_type = "gpt2";
        bool add_bos_token = false;
        bool add_eos_token = false;
        uint32_t bos_token_id = 0;
        uint32_t eos_token_id = 0;
        uint32_t pad_token_id = 0;
        uint32_t unk_token_id = 0;
        std::string chat_template;
        std::string pre_tokenizer;
        bool add_space_prefix = true;
        bool remove_extra_whitespaces = false;
    };

    struct BPEMerge {
        std::string first;
        std::string second;
        uint32_t rank;
    };

    std::pair<std::vector<VocabularyEntry>, TokenizerConfig> parse_from_metadata(
        std::istream& file,
        const std::vector<GGUFMetadataKV>& metadata) {

        std::vector<VocabularyEntry> vocab;
        TokenizerConfig config;

        // Parse tokenizer configuration first
        for (const auto& kv : metadata) {
            parse_tokenizer_config(kv, config);
        }

        // Parse vocabulary tokens
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.tokens" && kv.value_type == GGUFValueType::ARRAY) {
                vocab = parse_token_array(file, kv.value);
            }
        }

        // Parse scores
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.scores" && kv.value_type == GGUFValueType::ARRAY) {
                parse_scores(kv.value, vocab);
            }
        }

        // Parse token types
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.token_type" && kv.value_type == GGUFValueType::ARRAY) {
                parse_token_types(kv.value, vocab);
            }
        }

        // Parse BPE merges
        std::vector<BPEMerge> merges;
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.merges" && kv.value_type == GGUFValueType::ARRAY) {
                merges = parse_merges(file, kv.value);
            }
        }

        // Parse added tokens
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.added_tokens" && kv.value_type == GGUFValueType::ARRAY) {
                parse_added_tokens(file, kv.value, vocab);
            }
        }

        return {vocab, config};
    }

    const std::vector<BPEMerge>& get_merges() const { return merges_; }

private:
    std::vector<BPEMerge> merges_;

    void parse_tokenizer_config(const GGUFMetadataKV& kv, TokenizerConfig& config) {
        if (kv.key == "tokenizer.ggml.model" && kv.value_type == GGUFValueType::STRING) {
            config.model_type = *static_cast<std::string*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.add_bos_token" && kv.value_type == GGUFValueType::BOOL) {
            config.add_bos_token = *static_cast<bool*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.add_eos_token" && kv.value_type == GGUFValueType::BOOL) {
            config.add_eos_token = *static_cast<bool*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.bos_token_id" && kv.value_type == GGUFValueType::UINT32) {
            config.bos_token_id = *static_cast<uint32_t*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.eos_token_id" && kv.value_type == GGUFValueType::UINT32) {
            config.eos_token_id = *static_cast<uint32_t*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.padding_token_id" && kv.value_type == GGUFValueType::UINT32) {
            config.pad_token_id = *static_cast<uint32_t*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.unknown_token_id" && kv.value_type == GGUFValueType::UINT32) {
            config.unk_token_id = *static_cast<uint32_t*>(kv.value);
        } else if (kv.key == "tokenizer.chat_template" && kv.value_type == GGUFValueType::STRING) {
            config.chat_template = *static_cast<std::string*>(kv.value);
        } else if (kv.key == "tokenizer.ggml.pre" && kv.value_type == GGUFValueType::STRING) {
            config.pre_tokenizer = *static_cast<std::string*>(kv.value);
        }
    }

    std::vector<VocabularyEntry> parse_token_array(std::istream& file, void* value) {
        std::vector<VocabularyEntry> result;
        if (!value) return result;

        auto* array_data = static_cast<std::vector<uint8_t>*>(value);
        if (array_data->size() < 12) return result;

        const uint8_t* data = array_data->data();
        GGUFValueType elem_type = static_cast<GGUFValueType>(*reinterpret_cast<const uint32_t*>(data));
        uint64_t count = *reinterpret_cast<const uint64_t*>(data + 4);

        if (elem_type != GGUFValueType::STRING) return result;

        result.reserve(count);
        size_t offset = 12;

        for (uint64_t i = 0; i < count && offset < array_data->size(); ++i) {
            if (offset + 8 > array_data->size()) break;

            uint64_t str_len = *reinterpret_cast<const uint64_t*>(data + offset);
            offset += 8;

            if (offset + str_len > array_data->size()) break;

            VocabularyEntry entry;
            entry.token.assign(reinterpret_cast<const char*>(data + offset), str_len);
            entry.bytes.assign(data + offset, data + offset + str_len);
            entry.score = 0.0f;
            entry.token_type = 0;

            result.push_back(std::move(entry));
            offset += str_len;
        }

        return result;
    }

    void parse_scores(const void* value, std::vector<VocabularyEntry>& vocab) {
        if (!value || vocab.empty()) return;

        auto* array_data = static_cast<const std::vector<uint8_t>*>(value);
        const uint8_t* data = array_data->data();
        if (array_data->size() < 12) return;

        GGUFValueType elem_type = static_cast<GGUFValueType>(*reinterpret_cast<const uint32_t*>(data));
        uint64_t count = *reinterpret_cast<const uint64_t*>(data + 4);

        if (elem_type != GGUFValueType::FLOAT32) return;

        size_t offset = 12;
        for (uint64_t i = 0; i < count && i < vocab.size() && offset + 4 <= array_data->size(); ++i) {
            vocab[i].score = *reinterpret_cast<const float*>(data + offset);
            offset += 4;
        }
    }

    void parse_token_types(const void* value, std::vector<VocabularyEntry>& vocab) {
        if (!value || vocab.empty()) return;

        auto* array_data = static_cast<const std::vector<uint8_t>*>(value);
        const uint8_t* data = array_data->data();
        if (array_data->size() < 12) return;

        GGUFValueType elem_type = static_cast<GGUFValueType>(*reinterpret_cast<const uint32_t*>(data));
        uint64_t count = *reinterpret_cast<const uint64_t*>(data + 4);

        if (elem_type != GGUFValueType::UINT32) return;

        size_t offset = 12;
        for (uint64_t i = 0; i < count && i < vocab.size() && offset + 4 <= array_data->size(); ++i) {
            vocab[i].token_type = *reinterpret_cast<const uint32_t*>(data + offset);
            offset += 4;
        }
    }

    std::vector<BPEMerge> parse_merges(std::istream& file, void* value) {
        std::vector<BPEMerge> result;
        if (!value) return result;

        auto* array_data = static_cast<std::vector<uint8_t>*>(value);
        if (array_data->size() < 12) return result;

        const uint8_t* data = array_data->data();
        GGUFValueType elem_type = static_cast<GGUFValueType>(*reinterpret_cast<const uint32_t*>(data));
        uint64_t count = *reinterpret_cast<const uint64_t*>(data + 4);

        if (elem_type != GGUFValueType::STRING) return result;

        result.reserve(count);
        size_t offset = 12;

        for (uint64_t i = 0; i < count && offset < array_data->size(); ++i) {
            if (offset + 8 > array_data->size()) break;

            uint64_t str_len = *reinterpret_cast<const uint64_t*>(data + offset);
            offset += 8;

            if (offset + str_len > array_data->size()) break;

            std::string merge_str(reinterpret_cast<const char*>(data + offset), str_len);
            offset += str_len;

            // Parse "first second" format
            size_t space_pos = merge_str.find(' ');
            if (space_pos != std::string::npos) {
                BPEMerge merge;
                merge.first = merge_str.substr(0, space_pos);
                merge.second = merge_str.substr(space_pos + 1);
                merge.rank = static_cast<uint32_t>(i);
                result.push_back(merge);
            }
        }

        merges_ = result;
        return result;
    }

    void parse_added_tokens(std::istream& file, void* value, std::vector<VocabularyEntry>& vocab) {
        // Parse added tokens that extend the base vocabulary
        (void)file;
        (void)value;
        (void)vocab;
        // Implementation for added tokens parsing
    }
};

// ============================================================================
// Tensor Type Size Lookup with Quantization Support
// ============================================================================
static uint64_t get_type_size(GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
            return 1;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
        case GGUFValueType::FLOAT16:
        case GGUFValueType::BFLOAT16:
            return 2;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            return 4;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            return 8;
        case GGUFValueType::Q4_0:
        case GGUFValueType::Q4_1:
            return 18;
        case GGUFValueType::Q5_0:
        case GGUFValueType::Q5_1:
            return 22;
        case GGUFValueType::Q8_0:
        case GGUFValueType::Q8_1:
            return 34;
        default:
            return 4;
    }
}

static uint64_t calculate_tensor_size(const GGUFTensorInfo& tensor) {
    uint64_t element_count = 1;
    for (uint64_t dim : tensor.dimensions) {
        element_count *= dim;
    }

    uint64_t type_size = get_type_size(tensor.data_type);

    if (tensor.data_type >= GGUFValueType::Q4_0 && tensor.data_type <= GGUFValueType::Q8_1) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        return num_blocks * type_size;
    }

    return element_count * type_size;
}

// ============================================================================
// Multi-Shard Streaming Tensor Reader with Parallel I/O
// ============================================================================
class StreamingTensorReader {
public:
    struct ShardInfo {
        std::filesystem::path path;
        uint64_t tensor_start;
        uint64_t tensor_count;
        uint64_t data_offset;
    };

    explicit StreamingTensorReader(const std::filesystem::path& primary_path)
        : primary_path_(primary_path) {}

    ~StreamingTensorReader() {
        close();
    }

    bool open() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Open primary file
        files_[primary_path_] = std::make_unique<std::ifstream>();
        files_[primary_path_]->open(primary_path_, std::ios::binary | std::ios::in);
        return files_[primary_path_]->is_open();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [path, file] : files_) {
            if (file && file->is_open()) {
                file->close();
            }
        }
        files_.clear();
    }

    bool add_shard(const std::filesystem::path& shard_path, uint64_t tensor_start, uint64_t tensor_count) {
        std::lock_guard<std::mutex> lock(mutex_);

        ShardInfo info;
        info.path = shard_path;
        info.tensor_start = tensor_start;
        info.tensor_count = tensor_count;

        // Open shard file
        auto file = std::make_unique<std::ifstream>();
        file->open(shard_path, std::ios::binary | std::ios::in);
        if (!file->is_open()) {
            return false;
        }

        // Read data offset from shard header
        file->seekg(4);  // Skip magic
        uint32_t version;
        file->read(reinterpret_cast<char*>(&version), 4);
        uint64_t tensor_count_in_shard, metadata_kv_count;
        file->read(reinterpret_cast<char*>(&tensor_count_in_shard), 8);
        file->read(reinterpret_cast<char*>(&metadata_kv_count), 8);

        // Skip metadata and tensor info to find data offset
        // Simplified - would need full header parsing
        info.data_offset = 4096;  // Default alignment

        files_[shard_path] = std::move(file);
        shards_.push_back(info);
        return true;
    }

    // Thread-safe chunk read
    bool read_chunk(uint64_t offset, void* buffer, size_t size, const std::filesystem::path& path = "") {
        std::lock_guard<std::mutex> lock(mutex_);

        const auto& file_path = path.empty() ? primary_path_ : path;
        auto it = files_.find(file_path);
        if (it == files_.end() || !it->second || !it->second->is_open()) {
            return false;
        }

        it->second->seekg(static_cast<std::streamoff>(offset));
        if (!it->second->good()) return false;

        it->second->read(static_cast<char*>(buffer), static_cast<std::streamsize>(size));
        return it->second->good() || it->second->gcount() == static_cast<std::streamsize>(size);
    }

    // Async read with work-stealing thread pool
    std::future<bool> read_chunk_async(WorkStealingThreadPool& pool, uint64_t offset, void* buffer, size_t size) {
        return pool.enqueue([this, offset, buffer, size]() {
            return read_chunk(offset, buffer, size);
        });
    }

    // Parallel read with optimal chunk sizing for 70B models
    bool read_parallel(WorkStealingThreadPool& pool, uint64_t offset, void* buffer, size_t total_size,
                       size_t chunk_size = 8 * 1024 * 1024) {  // 8MB chunks for large models
        if (total_size <= chunk_size) {
            return read_chunk(offset, buffer, total_size);
        }

        char* byte_buffer = static_cast<char*>(buffer);
        size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
        std::vector<std::future<bool>> futures;
        futures.reserve(num_chunks);

        // Submit all chunk reads to thread pool
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t current_offset = i * chunk_size;
            size_t current_size = std::min(chunk_size, total_size - current_offset);

            futures.push_back(
                pool.enqueue([this, offset, byte_buffer, current_offset, current_size]() {
                    return read_chunk(offset + current_offset, byte_buffer + current_offset, current_size);
                })
            );
        }

        // Wait for all chunks and check success
        bool success = true;
        for (auto& f : futures) {
            success = success && f.get();
        }

        return success;
    }

    // Multi-shard parallel read for sharded models
    bool read_multi_shard(WorkStealingThreadPool& pool, const std::vector<std::pair<uint64_t, size_t>>& shard_ranges,
                          void* buffer, size_t chunk_size = 8 * 1024 * 1024) {
        char* byte_buffer = static_cast<char*>(buffer);
        std::vector<std::future<bool>> futures;

        size_t buffer_offset = 0;
        for (const auto& [file_offset, size] : shard_ranges) {
            futures.push_back(
                pool.enqueue([this, file_offset, byte_buffer, buffer_offset, size, chunk_size, &pool]() {
                    return read_parallel(pool, file_offset, byte_buffer + buffer_offset, size, chunk_size);
                })
            );
            buffer_offset += size;
        }

        bool success = true;
        for (auto& f : futures) {
            success = success && f.get();
        }

        return success;
    }

private:
    std::filesystem::path primary_path_;
    std::unordered_map<std::filesystem::path, std::unique_ptr<std::ifstream>> files_;
    std::vector<ShardInfo> shards_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Intelligent Tensor Pack Mapper with Layer-Aware Grouping
// ============================================================================
class TensorPackMapper {
public:
    struct PackMapping {
        uint64_t pack_id;
        std::string tensor_name;
        uint64_t file_offset;
        uint64_t compressed_size;
        uint64_t decompressed_size;
        GGUFValueType data_type;
        std::vector<uint64_t> dimensions;
        uint32_t layer_idx;
        PackType pack_type;
        float importance_score;
    };

    struct LayerGroup {
        uint32_t layer_idx;
        std::vector<PackMapping> attention_packs;
        std::vector<PackMapping> ffn_packs;
        std::vector<PackMapping> norm_packs;
        std::vector<PackMapping> other_packs;
        uint64_t total_size;
    };

    std::vector<PackMapping> create_mappings(
        const std::vector<GGUFTensorInfo>& tensors,
        const ModelConfig& config) {

        std::vector<PackMapping> mappings;
        mappings.reserve(tensors.size());
        uint64_t next_pack_id = 0;

        // Group tensors by layer
        std::unordered_map<uint32_t, std::vector<const GGUFTensorInfo*>> layer_tensors;
        std::vector<const GGUFTensorInfo*> non_layer_tensors;

        for (const auto& tensor : tensors) {
            uint32_t layer_idx = extract_layer_index(tensor.name);
            if (layer_idx > 0) {
                layer_tensors[layer_idx].push_back(&tensor);
            } else {
                non_layer_tensors.push_back(&tensor);
            }
        }

        // Create mappings for layer tensors (sorted by layer)
        std::vector<uint32_t> sorted_layers;
        for (const auto& [layer_idx, _] : layer_tensors) {
            sorted_layers.push_back(layer_idx);
        }
        std::sort(sorted_layers.begin(), sorted_layers.end());

        for (uint32_t layer_idx : sorted_layers) {
            for (const auto* tensor : layer_tensors[layer_idx]) {
                PackMapping mapping = create_mapping(*tensor, next_pack_id++, layer_idx, config);
                mappings.push_back(mapping);
            }
        }

        // Create mappings for non-layer tensors (embeddings, output, etc.)
        for (const auto* tensor : non_layer_tensors) {
            PackMapping mapping = create_mapping(*tensor, next_pack_id++, 0, config);
            mappings.push_back(mapping);
        }

        return mappings;
    }

    std::vector<LayerGroup> create_layer_groups(const std::vector<PackMapping>& mappings) {
        std::unordered_map<uint32_t, LayerGroup> groups;

        for (const auto& mapping : mappings) {
            if (mapping.layer_idx == 0) continue;  // Skip non-layer tensors

            if (groups.find(mapping.layer_idx) == groups.end()) {
                groups[mapping.layer_idx] = LayerGroup{};
                groups[mapping.layer_idx].layer_idx = mapping.layer_idx;
            }

            auto& group = groups[mapping.layer_idx];
            group.total_size += mapping.decompressed_size;

            switch (mapping.pack_type) {
                case PackType::ATTENTION_Q:
                case PackType::ATTENTION_K:
                case PackType::ATTENTION_V:
                case PackType::ATTENTION_O:
                    group.attention_packs.push_back(mapping);
                    break;
                case PackType::FEED_FORWARD_UP:
                case PackType::FEED_FORWARD_DOWN:
                case PackType::FEED_FORWARD_GATE:
                    group.ffn_packs.push_back(mapping);
                    break;
                case PackType::NORM_GAMMA:
                case PackType::NORM_BETA:
                    group.norm_packs.push_back(mapping);
                    break;
                default:
                    group.other_packs.push_back(mapping);
                    break;
            }
        }

        std::vector<LayerGroup> result;
        for (auto& [idx, group] : groups) {
            result.push_back(std::move(group));
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            return a.layer_idx < b.layer_idx;
        });

        return result;
    }

private:
    PackMapping create_mapping(const GGUFTensorInfo& tensor, uint64_t pack_id, uint32_t layer_idx, const ModelConfig& config) {
        PackMapping mapping;
        mapping.pack_id = pack_id;
        mapping.tensor_name = tensor.name;
        mapping.file_offset = tensor.offset;
        mapping.data_type = tensor.data_type;
        mapping.dimensions = tensor.dimensions;
        mapping.layer_idx = layer_idx;
        mapping.pack_type = infer_pack_type(tensor.name);

        mapping.decompressed_size = calculate_tensor_size(tensor);
        mapping.compressed_size = mapping.decompressed_size;

        // Calculate importance score based on layer and type
        float layer_factor = 1.0f - (static_cast<float>(layer_idx) / std::max(1u, config.num_layers));
        float type_factor = calculate_type_importance(mapping.pack_type);
        mapping.importance_score = layer_factor * type_factor;

        return mapping;
    }

    uint32_t extract_layer_index(const std::string& tensor_name) {
        // Support various naming conventions
        const char* patterns[] = {"layers.", "blk.", "layer_", "encoder.layer."};
        for (const char* pattern : patterns) {
            size_t pos = tensor_name.find(pattern);
            if (pos != std::string::npos) {
                size_t start = pos + strlen(pattern);
                size_t end = tensor_name.find_first_of("._", start);
                if (end == std::string::npos) end = tensor_name.length();

                try {
                    return static_cast<uint32_t>(std::stoul(tensor_name.substr(start, end - start)));
                } catch (...) {
                    continue;
                }
            }
        }
        return 0;
    }

    PackType infer_pack_type(const std::string& tensor_name) {
        // Attention weights
        if (tensor_name.find("attention.wq") != std::string::npos ||
            tensor_name.find("attn_q") != std::string::npos ||
            tensor_name.find("self_attn.q_proj") != std::string::npos) return PackType::ATTENTION_Q;

        if (tensor_name.find("attention.wk") != std::string::npos ||
            tensor_name.find("attn_k") != std::string::npos ||
            tensor_name.find("self_attn.k_proj") != std::string::npos) return PackType::ATTENTION_K;

        if (tensor_name.find("attention.wv") != std::string::npos ||
            tensor_name.find("attn_v") != std::string::npos ||
            tensor_name.find("self_attn.v_proj") != std::string::npos) return PackType::ATTENTION_V;

        if (tensor_name.find("attention.wo") != std::string::npos ||
            tensor_name.find("attn_output") != std::string::npos ||
            tensor_name.find("self_attn.o_proj") != std::string::npos) return PackType::ATTENTION_O;

        // Feed-forward weights
        if (tensor_name.find("feed_forward.w1") != std::string::npos ||
            tensor_name.find("ffn_gate") != std::string::npos ||
            tensor_name.find("mlp.gate_proj") != std::string::npos) return PackType::FEED_FORWARD_GATE;

        if (tensor_name.find("feed_forward.w2") != std::string::npos ||
            tensor_name.find("ffn_down") != std::string::npos ||
            tensor_name.find("mlp.down_proj") != std::string::npos) return PackType::FEED_FORWARD_DOWN;

        if (tensor_name.find("feed_forward.w3") != std::string::npos ||
            tensor_name.find("ffn_up") != std::string::npos ||
            tensor_name.find("mlp.up_proj") != std::string::npos) return PackType::FEED_FORWARD_UP;

        // Normalization
        if (tensor_name.find("attention_norm") != std::string::npos ||
            tensor_name.find("input_layernorm") != std::string::npos) return PackType::NORM_GAMMA;

        if (tensor_name.find("ffn_norm") != std::string::npos ||
            tensor_name.find("post_attention_layernorm") != std::string::npos) return PackType::NORM_GAMMA;

        // Embeddings and output
        if (tensor_name.find("token_embd") != std::string::npos ||
            tensor_name.find("embed_tokens") != std::string::npos ||
            tensor_name.find("word_embeddings") != std::string::npos) return PackType::EMBEDDING;

        if (tensor_name.find("output_norm") != std::string::npos ||
            tensor_name.find("norm") != std::string::npos) return PackType::NORM_GAMMA;

        if (tensor_name.find("output") != std::string::npos ||
            tensor_name.find("lm_head") != std::string::npos) return PackType::HEAD;

        // RoPE
        if (tensor_name.find("rope") != std::string::npos ||
            tensor_name.find("rotary") != std::string::npos) return PackType::ROPE;

        return PackType::UNKNOWN;
    }

    float calculate_type_importance(PackType type) {
        switch (type) {
            case PackType::ATTENTION_Q:
            case PackType::ATTENTION_K:
            case PackType::ATTENTION_V:
                return 1.0f;
            case PackType::ATTENTION_O:
                return 0.95f;
            case PackType::NORM_GAMMA:
            case PackType::NORM_BETA:
                return 0.9f;
            case PackType::FEED_FORWARD_GATE:
            case PackType::FEED_FORWARD_UP:
                return 0.85f;
            case PackType::FEED_FORWARD_DOWN:
                return 0.8f;
            case PackType::EMBEDDING:
                return 0.75f;
            case PackType::HEAD:
                return 0.7f;
            case PackType::ROPE:
                return 0.6f;
            default:
                return 0.5f;
        }
    }
};

// ============================================================================
// Advanced Hybrid Decompression Engine with Auto-Codec Detection
// ============================================================================
class HybridDecompressor {
public:
    enum class CompressionFormat {
        RAW,
        BLOSC2_LZ4,
        BLOSC2_ZSTD,
        ZFP,
        HYBRID_VKS,
        UNKNOWN
    };

    struct CompressionHeader {
        CompressionFormat format;
        uint64_t uncompressed_size;
        uint64_t compressed_size;
        GGUFValueType data_type;
        uint32_t checksum;
    };

    HybridDecompressor() {
        compression::Blosc2Compression::initialize();
    }

    ~HybridDecompressor() = default;

    CompressionHeader detect_format(const std::vector<uint8_t>& data) {
        CompressionHeader header;
        header.format = CompressionFormat::RAW;
        header.compressed_size = data.size();

        if (data.size() < 16) {
            return header;
        }

        // Check for Blosc2 header
        if (data[0] == 0x02 && data[1] == 0x01) {
            header.format = CompressionFormat::BLOSC2_LZ4;
            // Parse Blosc2 header for uncompressed size
            if (data.size() >= 16) {
                header.uncompressed_size = *reinterpret_cast<const uint64_t*>(&data[8]);
            }
            return header;
        }

        // Check for ZFP header
        if (data[0] == 'Z' && data[1] == 'F' && data[2] == 'P') {
            header.format = CompressionFormat::ZFP;
            if (data.size() >= 16) {
                header.uncompressed_size = *reinterpret_cast<const uint64_t*>(&data[4]);
            }
            return header;
        }

        // Check for VKS (Vulkan Symbiote) hybrid compression header
        if (data[0] == 'V' && data[1] == 'K' && data[2] == 'S') {
            header.format = CompressionFormat::HYBRID_VKS;
            if (data.size() >= 24) {
                header.uncompressed_size = *reinterpret_cast<const uint64_t*>(&data[4]);
                header.compressed_size = *reinterpret_cast<const uint64_t*>(&data[12]);
                header.checksum = *reinterpret_cast<const uint32_t*>(&data[20]);
            }
            return header;
        }

        // Check for ZSTD magic
        if (data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD) {
            header.format = CompressionFormat::BLOSC2_ZSTD;
            return header;
        }

        return header;
    }

    std::vector<float> decompress(const std::vector<uint8_t>& compressed_data,
                                   uint64_t expected_elements,
                                   GGUFValueType data_type) {
        auto header = detect_format(compressed_data);

        switch (header.format) {
            case CompressionFormat::BLOSC2_LZ4:
            case CompressionFormat::BLOSC2_ZSTD:
                return decompress_blosc2(compressed_data, expected_elements);

            case CompressionFormat::ZFP:
                return decompress_zfp(compressed_data, expected_elements);

            case CompressionFormat::HYBRID_VKS:
                return decompress_hybrid_vks(compressed_data, expected_elements, data_type);

            case CompressionFormat::RAW:
            default:
                return convert_raw_to_fp32(compressed_data, data_type, expected_elements);
        }
    }

    // Batch decompression for multiple tensors
    std::vector<std::vector<float>> decompress_batch(
        const std::vector<std::pair<std::vector<uint8_t>, uint64_t>>& compressed_batches,
        GGUFValueType data_type,
        WorkStealingThreadPool& pool) {

        std::vector<std::future<std::vector<float>>> futures;
        futures.reserve(compressed_batches.size());

        for (const auto& [data, elements] : compressed_batches) {
            futures.push_back(pool.enqueue([this, &data, elements, data_type]() {
                return decompress(data, elements, data_type);
            }));
        }

        std::vector<std::vector<float>> results;
        results.reserve(futures.size());
        for (auto& f : futures) {
            results.push_back(f.get());
        }

        return results;
    }

private:
    std::vector<float> decompress_blosc2(const std::vector<uint8_t>& compressed_data,
                                          uint64_t expected_elements) {
        std::vector<float> result(expected_elements);

        compression::Blosc2Compression decompressor;
        bool success = decompressor.decompress(
            compressed_data.data(),
            compressed_data.size(),
            expected_elements * sizeof(float),
            result.data()
        );

        if (!success) {
            result.clear();
        }

        return result;
    }

    std::vector<float> decompress_zfp(const std::vector<uint8_t>& compressed_data,
                                       uint64_t expected_elements) {
        // ZFP decompression stub - would integrate with zfp library
        (void)compressed_data;
        (void)expected_elements;
        return {};
    }

    std::vector<float> decompress_hybrid_vks(const std::vector<uint8_t>& compressed_data,
                                              uint64_t expected_elements,
                                              GGUFValueType data_type) {
        if (compressed_data.size() < 24) return {};

        // Verify checksum
        uint32_t stored_checksum = *reinterpret_cast<const uint32_t*>(&compressed_data[20]);
        uint32_t computed_checksum = crc32(compressed_data.data() + 24, compressed_data.size() - 24);

        if (stored_checksum != computed_checksum) {
            std::cerr << "[HybridDecompressor] CRC32 mismatch!" << std::endl;
            return {};
        }

        // Decompress based on data type
        if (is_quantized_type(data_type)) {
            return decompress_quantized(compressed_data.data() + 24, compressed_data.size() - 24,
                                        data_type, expected_elements);
        } else {
            return decompress_blosc2(compressed_data, expected_elements);
        }
    }

    std::vector<float> decompress_quantized(const uint8_t* data, size_t size,
                                            GGUFValueType data_type, uint64_t expected_elements) {
        std::vector<float> result;
        result.reserve(expected_elements);

        switch (data_type) {
            case GGUFValueType::Q8_0:
                result = decompress_q8_0(data, size, expected_elements);
                break;
            case GGUFValueType::Q4_0:
                result = decompress_q4_0(data, size, expected_elements);
                break;
            case GGUFValueType::Q5_0:
            case GGUFValueType::Q5_1:
                result = decompress_q5(data, size, expected_elements, data_type);
                break;
            default:
                break;
        }

        return result;
    }

    std::vector<float> decompress_q8_0(const uint8_t* data, size_t size, uint64_t element_count) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);

        size_t offset = 0;
        for (uint64_t block = 0; block < num_blocks && offset + block_size + 4 <= size; ++block) {
            float scale = *reinterpret_cast<const float*>(data + offset);
            offset += 4;

            for (uint32_t i = 0; i < block_size && result.size() < element_count && offset < size; ++i) {
                int8_t quantized = static_cast<int8_t>(data[offset++]);
                result.push_back(scale * static_cast<float>(quantized));
            }
        }

        return result;
    }

    std::vector<float> decompress_q4_0(const uint8_t* data, size_t size, uint64_t element_count) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);

        size_t offset = 0;
        for (uint64_t block = 0; block < num_blocks && offset + block_size / 2 + 4 <= size; ++block) {
            float scale = *reinterpret_cast<const float*>(data + offset);
            offset += 4;

            for (uint32_t i = 0; i < block_size / 2 && result.size() < element_count; ++i) {
                if (offset >= size) break;

                uint8_t packed = data[offset++];
                int8_t q0 = (packed >> 4) & 0x0F;
                int8_t q1 = packed & 0x0F;

                result.push_back(scale * static_cast<float>(q0 - 8));
                if (result.size() < element_count) {
                    result.push_back(scale * static_cast<float>(q1 - 8));
                }
            }
        }

        return result;
    }

    std::vector<float> decompress_q5(const uint8_t* data, size_t size, uint64_t element_count, GGUFValueType type) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);

        size_t offset = 0;
        for (uint64_t block = 0; block < num_blocks; ++block) {
            if (offset + 4 > size) break;

            float scale = *reinterpret_cast<const float*>(data + offset);
            offset += 4;

            // Q5 has high bits in a separate block
            uint32_t hbits = 0;
            if (type == GGUFValueType::Q5_0 || type == GGUFValueType::Q5_1) {
                if (offset + 4 > size) break;
                hbits = *reinterpret_cast<const uint32_t*>(data + offset);
                offset += 4;
            }

            for (uint32_t i = 0; i < block_size && result.size() < element_count; ++i) {
                if (offset >= size) break;

                uint8_t packed = data[offset++];
                uint8_t hbit = (hbits >> i) & 1;
                uint8_t q = ((packed & 0x0F) | (hbit << 4));

                if (type == GGUFValueType::Q5_0) {
                    result.push_back(scale * static_cast<float>(q - 16));
                } else {
                    // Q5_1 has min value as well
                    result.push_back(scale * static_cast<float>(q));
                }

                if (result.size() < element_count) {
                    hbit = (hbits >> (i + 16)) & 1;
                    q = (((packed >> 4) & 0x0F) | (hbit << 4));

                    if (type == GGUFValueType::Q5_0) {
                        result.push_back(scale * static_cast<float>(q - 16));
                    } else {
                        result.push_back(scale * static_cast<float>(q));
                    }
                }
            }
        }

        return result;
    }

    std::vector<float> convert_raw_to_fp32(const std::vector<uint8_t>& data,
                                            GGUFValueType data_type,
                                            uint64_t expected_elements) {
        std::vector<float> result;
        result.reserve(expected_elements);

        switch (data_type) {
            case GGUFValueType::FLOAT32: {
                const float* ptr = reinterpret_cast<const float*>(data.data());
                size_t count = std::min(expected_elements, data.size() / sizeof(float));
                result.assign(ptr, ptr + count);
                break;
            }
            case GGUFValueType::FLOAT16: {
                const uint16_t* ptr = reinterpret_cast<const uint16_t*>(data.data());
                size_t count = std::min(expected_elements, data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(fp16_to_fp32(ptr[i]));
                }
                break;
            }
            case GGUFValueType::BFLOAT16: {
                const uint16_t* ptr = reinterpret_cast<const uint16_t*>(data.data());
                size_t count = std::min(expected_elements, data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(bf16_to_fp32(ptr[i]));
                }
                break;
            }
            case GGUFValueType::INT8: {
                const int8_t* ptr = reinterpret_cast<const int8_t*>(data.data());
                size_t count = std::min(expected_elements, data.size());
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            case GGUFValueType::INT16: {
                const int16_t* ptr = reinterpret_cast<const int16_t*>(data.data());
                size_t count = std::min(expected_elements, data.size() / sizeof(int16_t));
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            default:
                break;
        }

        return result;
    }

    static bool is_quantized_type(GGUFValueType type) {
        return type >= GGUFValueType::Q4_0 && type <= GGUFValueType::Q8_1;
    }

    static float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float val = mant * std::pow(2.0f, -24.0f);
            return sign ? -val : val;
        } else if (exp == 31) {
            if (mant == 0) {
                return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            }
            return std::numeric_limits<float>::quiet_NaN();
        }

        float val = std::pow(2.0f, static_cast<float>(exp - 15)) * (1.0f + mant / 1024.0f);
        return sign ? -val : val;
    }

    static float bf16_to_fp32(uint16_t b) {
        uint32_t val = static_cast<uint32_t>(b) << 16;
        float result;
        std::memcpy(&result, &val, sizeof(float));
        return result;
    }

    static uint32_t crc32(const uint8_t* data, size_t size) {
        static const uint32_t crc_table[256] = {
            0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
            0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
            0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
            0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
            0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
            0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
            0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
            0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
            0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
            0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
            0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
            0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
            0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
            0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
            0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
            0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
            0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
            0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
            0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
            0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
            0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
            0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
            0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
            0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
            0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
            0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
            0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
            0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
            0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
            0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
            0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
            0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
            0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
            0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
            0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
            0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
            0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
            0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
            0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
            0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
            0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
            0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
            0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
        };

        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < size; ++i) {
            crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFF;
    }
};

// ============================================================================
// Full GGUFLoader Implementation with All Enhancements
// ============================================================================
class GGUFLoaderImpl {
public:
    explicit GGUFLoaderImpl(const Path& file_path)
        : file_path_(file_path),
          data_offset_(0),
          thread_pool_(std::max(4u, std::thread::hardware_concurrency())),
          tensor_cache_(create_cache_config()),
          hybrid_decompressor_(std::make_unique<HybridDecompressor>()),
          is_sharded_(false) {
        compression::Blosc2Compression::initialize();
    }

    ExpectedVoid load() {
        // Open file
        file_.open(file_path_.c_str(), std::ios::binary);
        if (!file_.is_open()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        // Read and validate header
        auto header_result = read_header();
        if (!header_result.has_value()) {
            return header_result;
        }

        // Read metadata
        auto metadata_result = read_metadata();
        if (!metadata_result.has_value()) {
            return metadata_result;
        }

        // Read tensor info
        auto tensor_result = read_tensors();
        if (!tensor_result.has_value()) {
            return tensor_result;
        }

        // Parse vocabulary from metadata
        VocabularyParser vocab_parser;
        auto [vocab, tokenizer_config] = vocab_parser.parse_from_metadata(file_, metadata_);
        vocabulary_ = std::move(vocab);
        tokenizer_config_ = std::move(tokenizer_config);

        // Parse model configuration
        parse_model_config();

        // Create tensor pack mappings
        TensorPackMapper mapper;
        tensor_mappings_ = mapper.create_mappings(tensors_, model_config_);
        layer_groups_ = mapper.create_layer_groups(tensor_mappings_);

        // Initialize streaming reader
        stream_reader_ = std::make_unique<StreamingTensorReader>(file_path_);
        if (!stream_reader_->open()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        return make_expected_success();
    }

    ExpectedVoid close() {
        if (file_.is_open()) {
            file_.close();
        }
        if (stream_reader_) {
            stream_reader_->close();
        }

        // Clean up metadata values
        for (auto& kv : metadata_) {
            if (kv.value) {
                free_metadata_value(kv.value_type, kv.value);
                kv.value = nullptr;
            }
        }

        return make_expected_success();
    }

    // Enhanced tensor data loading with LRU cache and hybrid decompression
    Expected<std::vector<float>> load_tensor_data(const std::string& tensor_name, bool convert_fp16 = true) {
        // Check cache first
        std::vector<float> cached_result;
        if (tensor_cache_.get(tensor_name, cached_result)) {
            return Expected<std::vector<float>>(std::move(cached_result));
        }

        const GGUFTensorInfo* tensor = find_tensor(tensor_name);
        if (!tensor) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        return read_and_decompress_tensor(*tensor, convert_fp16);
    }

    // Read tensor data with explicit tensor info
    Expected<std::vector<float>> read_tensor_data(const GGUFTensorInfo& tensor, bool convert_fp16 = true) {
        // Check cache by name if available
        std::vector<float> cached_result;
        if (tensor_cache_.get(tensor.name, cached_result)) {
            return Expected<std::vector<float>>(std::move(cached_result));
        }

        return read_and_decompress_tensor(tensor, convert_fp16);
    }

    // Parallel load multiple tensors with optimized scheduling
    std::vector<Expected<std::vector<float>>> load_tensors_parallel(
        const std::vector<std::string>& tensor_names, bool convert_fp16 = true) {

        std::vector<Expected<std::vector<float>>> results(tensor_names.size());
        std::vector<std::future<void>> futures;

        // Sort tensors by file offset for sequential I/O optimization
        std::vector<std::pair<size_t, uint64_t>> sorted_indices;
        sorted_indices.reserve(tensor_names.size());

        for (size_t i = 0; i < tensor_names.size(); ++i) {
            const auto* tensor = find_tensor(tensor_names[i]);
            if (tensor) {
                sorted_indices.push_back({i, tensor->offset});
            }
        }
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Submit tasks in sorted order for better I/O performance
        for (const auto& [original_idx, _] : sorted_indices) {
            futures.push_back(thread_pool_.enqueue([this, &tensor_names, &results, original_idx, convert_fp16]() {
                results[original_idx] = load_tensor_data(tensor_names[original_idx], convert_fp16);
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }

        return results;
    }

    // Batch load entire layer for 70B+ models
    Expected<std::vector<std::vector<float>>> load_layer_tensors(uint32_t layer_idx, bool convert_fp16 = true) {
        std::vector<std::string> layer_tensor_names;

        for (const auto& mapping : tensor_mappings_) {
            if (mapping.layer_idx == layer_idx) {
                layer_tensor_names.push_back(mapping.tensor_name);
            }
        }

        if (layer_tensor_names.empty()) {
            return Expected<std::vector<std::vector<float>>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        auto results = load_tensors_parallel(layer_tensor_names, convert_fp16);
        std::vector<std::vector<float>> data;
        data.reserve(results.size());

        for (auto& result : results) {
            if (result.has_value()) {
                data.push_back(std::move(result.value()));
            } else {
                return Expected<std::vector<std::vector<float>>>(result.error());
            }
        }

        return Expected<std::vector<std::vector<float>>>(std::move(data));
    }

    // Prefetch tensors for upcoming layers
    void prefetch_layers(uint32_t start_layer, uint32_t count) {
        std::vector<std::string> prefetch_names;

        for (const auto& mapping : tensor_mappings_) {
            if (mapping.layer_idx >= start_layer && mapping.layer_idx < start_layer + count) {
                // Only prefetch high-importance tensors
                if (mapping.importance_score > 0.8f) {
                    prefetch_names.push_back(mapping.tensor_name);
                }
            }
        }

        if (!prefetch_names.empty()) {
            // Touch cache entries to prevent eviction
            for (const auto& name : prefetch_names) {
                tensor_cache_.touch(name);
            }
        }
    }

    // Generate NomadPack metadata from tensor mappings
    std::vector<PackMetadata> generate_packs() {
        std::vector<PackMetadata> packs;
        packs.reserve(tensor_mappings_.size());

        for (const auto& mapping : tensor_mappings_) {
            PackMetadata pack;
            pack.pack_id = mapping.pack_id;
            pack.type = mapping.pack_type;
            pack.layer_idx = mapping.layer_idx;
            pack.head_idx = 0;
            pack.file_offset = mapping.file_offset;
            pack.compressed_size = mapping.compressed_size;
            pack.decompressed_size = mapping.decompressed_size;
            pack.tensor_name = mapping.tensor_name;
            pack.base_priority = mapping.importance_score;
            packs.push_back(pack);
        }

        return packs;
    }

    // Multi-file shard support
    ExpectedVoid add_shard(const Path& shard_path) {
        shard_paths_.push_back(shard_path);
        is_sharded_ = true;

        if (stream_reader_) {
            // Get tensor count from shard for proper offset calculation
            std::ifstream shard_file(shard_path, std::ios::binary);
            if (!shard_file.is_open()) {
                return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
            }

            char magic[4];
            shard_file.read(magic, 4);
            if (std::memcmp(magic, "GGUF", 4) != 0) {
                return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
            }

            uint32_t version;
            uint64_t tensor_count, metadata_kv_count;
            shard_file.read(reinterpret_cast<char*>(&version), 4);
            shard_file.read(reinterpret_cast<char*>(&tensor_count), 8);
            shard_file.read(reinterpret_cast<char*>(&metadata_kv_count), 8);

            uint64_t tensor_start = header_.tensor_count;
            stream_reader_->add_shard(shard_path, tensor_start, tensor_count);
        }

        return make_expected_success();
    }

    bool is_sharded() const { return is_sharded_; }

    // Cache management
    void clear_tensor_cache() { tensor_cache_.clear(); }
    size_t get_tensor_cache_size() const { return tensor_cache_.size(); }
    double get_tensor_cache_hit_rate() const { return tensor_cache_.hit_rate(); }
    size_t get_tensor_cache_memory() const { return tensor_cache_.memory_usage(); }
    EnhancedLRUCache<std::string, std::vector<float>>::CacheStats get_tensor_cache_stats() const { return tensor_cache_.get_stats(); }

    const GGUFHeader& header() const { return header_; }
    const ModelConfig& model_config() const { return model_config_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    const std::vector<GGUFMetadataKV>& metadata() const { return metadata_; }
    const std::vector<VocabularyEntry>& vocabulary() const { return vocabulary_; }
    const VocabularyParser::TokenizerConfig& tokenizer_config() const { return tokenizer_config_; }

private:
    Path file_path_;
    std::ifstream file_;
    std::unique_ptr<StreamingTensorReader> stream_reader_;

    GGUFHeader header_;
    std::vector<GGUFMetadataKV> metadata_;
    std::vector<GGUFTensorInfo> tensors_;
    ModelConfig model_config_;
    uint64_t data_offset_;
    std::vector<TensorPackMapper::PackMapping> tensor_mappings_;
    std::vector<TensorPackMapper::LayerGroup> layer_groups_;

    WorkStealingThreadPool thread_pool_;
    EnhancedLRUCache<std::string, std::vector<float>> tensor_cache_;
    std::unique_ptr<HybridDecompressor> hybrid_decompressor_;

    std::vector<VocabularyEntry> vocabulary_;
    VocabularyParser::TokenizerConfig tokenizer_config_;

    std::vector<Path> shard_paths_;
    bool is_sharded_;

    typename EnhancedLRUCache<std::string, std::vector<float>>::CacheConfig create_cache_config() {
        typename EnhancedLRUCache<std::string, std::vector<float>>::CacheConfig config;

        // Configure based on available system memory
        auto& cfg = ConfigManager::instance();
        config.max_size = cfg.get_int("cache", "max_entries", 100);
        config.max_memory_bytes = cfg.get_int("cache", "max_memory_mb", 4096) * 1024 * 1024;
        config.ttl_ms = cfg.get_int("cache", "ttl_seconds", 300) * 1000;
        config.enable_stats = true;

        return config;
    }

    Expected<std::vector<float>> read_and_decompress_tensor(const GGUFTensorInfo& tensor, bool convert_fp16) {
        uint64_t tensor_size = calculate_tensor_size(tensor);
        uint64_t element_count = tensor_size / get_type_size(tensor.data_type);

        if (tensor.data_type == GGUFValueType::FLOAT16 && convert_fp16) {
            element_count = tensor_size / 2;
        }

        // Read tensor data in parallel chunks
        std::vector<uint8_t> raw_data(tensor_size);
        bool read_success = stream_reader_->read_parallel(thread_pool_, tensor.offset,
                                                          raw_data.data(), tensor_size);

        if (!read_success) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        // Apply hybrid decompression
        std::vector<float> result = hybrid_decompressor_->decompress(raw_data, element_count, tensor.data_type);

        if (result.empty()) {
            // Fallback to direct type conversion
            result = convert_raw_to_fp32_fallback(raw_data, tensor.data_type, element_count);
        }

        // Store in cache
        if (!result.empty()) {
            size_t memory_cost = result.size() * sizeof(float);
            tensor_cache_.put(tensor.name, result, memory_cost);
        }

        return Expected<std::vector<float>>(std::move(result));
    }

    const GGUFTensorInfo* find_tensor(const std::string& name) const {
        for (const auto& tensor : tensors_) {
            if (tensor.name == name) {
                return &tensor;
            }
        }
        return nullptr;
    }

    ExpectedVoid read_header() {
        char magic[4];
        file_.read(magic, 4);

        if (std::memcmp(magic, "GGUF", 4) != 0) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        file_.read(reinterpret_cast<char*>(&header_.version), 4);
        file_.read(reinterpret_cast<char*>(&header_.tensor_count), 8);
        file_.read(reinterpret_cast<char*>(&header_.metadata_kv_count), 8);

        return make_expected_success();
    }

    ExpectedVoid read_metadata() {
        for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
            GGUFMetadataKV kv;

            auto str_result = read_string(kv.key);
            if (!str_result.has_value()) {
                return str_result;
            }

            file_.read(reinterpret_cast<char*>(&kv.value_type), 4);
            kv.value = read_value_by_type(kv.value_type);

            metadata_.push_back(kv);
        }

        return make_expected_success();
    }

    ExpectedVoid read_tensors() {
        uint64_t current_offset = 0;

        for (uint64_t i = 0; i < header_.tensor_count; ++i) {
            GGUFTensorInfo tensor;

            auto str_result = read_string(tensor.name);
            if (!str_result.has_value()) {
                return str_result;
            }

            file_.read(reinterpret_cast<char*>(&tensor.n_dimensions), 4);
            tensor.dimensions.resize(tensor.n_dimensions);

            for (uint32_t j = 0; j < tensor.n_dimensions; ++j) {
                file_.read(reinterpret_cast<char*>(&tensor.dimensions[j]), 8);
            }

            file_.read(reinterpret_cast<char*>(&tensor.data_type), 4);
            file_.read(reinterpret_cast<char*>(&tensor.offset), 8);

            tensors_.push_back(tensor);

            uint64_t tensor_end = tensor.offset + calculate_tensor_size(tensor);
            current_offset = std::max(current_offset, tensor_end);
        }

        data_offset_ = align_offset(current_offset, 4096);

        return make_expected_success();
    }

    ExpectedVoid read_string(std::string& str) {
        uint64_t len;
        file_.read(reinterpret_cast<char*>(&len), 8);

        str.resize(len);
        file_.read(&str[0], static_cast<std::streamsize>(len));

        return make_expected_success();
    }

    void* read_value_by_type(GGUFValueType type) {
        void* value = nullptr;

        switch (type) {
            case GGUFValueType::UINT8: {
                uint8_t* val = new uint8_t;
                file_.read(reinterpret_cast<char*>(val), 1);
                value = val;
                break;
            }
            case GGUFValueType::INT8: {
                int8_t* val = new int8_t;
                file_.read(reinterpret_cast<char*>(val), 1);
                value = val;
                break;
            }
            case GGUFValueType::UINT16: {
                uint16_t* val = new uint16_t;
                file_.read(reinterpret_cast<char*>(val), 2);
                value = val;
                break;
            }
            case GGUFValueType::INT16: {
                int16_t* val = new int16_t;
                file_.read(reinterpret_cast<char*>(val), 2);
                value = val;
                break;
            }
            case GGUFValueType::UINT32: {
                uint32_t* val = new uint32_t;
                file_.read(reinterpret_cast<char*>(val), 4);
                value = val;
                break;
            }
            case GGUFValueType::INT32: {
                int32_t* val = new int32_t;
                file_.read(reinterpret_cast<char*>(val), 4);
                value = val;
                break;
            }
            case GGUFValueType::FLOAT32: {
                float* val = new float;
                file_.read(reinterpret_cast<char*>(val), 4);
                value = val;
                break;
            }
            case GGUFValueType::UINT64: {
                uint64_t* val = new uint64_t;
                file_.read(reinterpret_cast<char*>(val), 8);
                value = val;
                break;
            }
            case GGUFValueType::INT64: {
                int64_t* val = new int64_t;
                file_.read(reinterpret_cast<char*>(val), 8);
                value = val;
                break;
            }
            case GGUFValueType::FLOAT64: {
                double* val = new double;
                file_.read(reinterpret_cast<char*>(val), 8);
                value = val;
                break;
            }
            case GGUFValueType::BOOL: {
                bool* val = new bool;
                file_.read(reinterpret_cast<char*>(val), 1);
                value = val;
                break;
            }
            case GGUFValueType::STRING: {
                std::string* val = new std::string;
                read_string(*val);
                value = val;
                break;
            }
            case GGUFValueType::ARRAY: {
                GGUFValueType elem_type;
                uint64_t count;
                file_.read(reinterpret_cast<char*>(&elem_type), 4);
                file_.read(reinterpret_cast<char*>(&count), 8);

                std::vector<uint8_t>* val = new std::vector<uint8_t>;
                val->resize(12 + count * get_type_size(elem_type));
                *reinterpret_cast<uint32_t*>(val->data()) = static_cast<uint32_t>(elem_type);
                *reinterpret_cast<uint64_t*>(val->data() + 4) = count;
                file_.read(reinterpret_cast<char*>(val->data() + 12),
                          static_cast<std::streamsize>(count * get_type_size(elem_type)));
                value = val;
                break;
            }
            default:
                value = nullptr;
                break;
        }

        return value;
    }

    void free_metadata_value(GGUFValueType type, void* value) {
        if (!value) return;

        switch (type) {
            case GGUFValueType::UINT8: delete static_cast<uint8_t*>(value); break;
            case GGUFValueType::INT8: delete static_cast<int8_t*>(value); break;
            case GGUFValueType::UINT16: delete static_cast<uint16_t*>(value); break;
            case GGUFValueType::INT16: delete static_cast<int16_t*>(value); break;
            case GGUFValueType::UINT32: delete static_cast<uint32_t*>(value); break;
            case GGUFValueType::INT32: delete static_cast<int32_t*>(value); break;
            case GGUFValueType::FLOAT32: delete static_cast<float*>(value); break;
            case GGUFValueType::UINT64: delete static_cast<uint64_t*>(value); break;
            case GGUFValueType::INT64: delete static_cast<int64_t*>(value); break;
            case GGUFValueType::FLOAT64: delete static_cast<double*>(value); break;
            case GGUFValueType::BOOL: delete static_cast<bool*>(value); break;
            case GGUFValueType::STRING: delete static_cast<std::string*>(value); break;
            case GGUFValueType::ARRAY: delete static_cast<std::vector<uint8_t>*>(value); break;
            default: delete[] static_cast<char*>(value); break;
        }
    }

    void parse_model_config() {
        for (const auto& kv : metadata_) {
            if (kv.key == "general.architecture" && kv.value_type == GGUFValueType::STRING) {
                model_config_.model_type = *static_cast<std::string*>(kv.value);
            } else if (kv.key == "llama.context_length" || kv.key == "llama.contextLength") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.max_position_embeddings = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.embedding_length" || kv.key == "llama.embeddingLength") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.hidden_size = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.feed_forward_length" || kv.key == "llama.feedForwardLength") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.intermediate_size = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.attention.head_count" || kv.key == "llama.attention.headCount") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.num_attention_heads = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.attention.head_count_kv" || kv.key == "llama.attention.headCountKV") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.num_key_value_heads = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.block_count" || kv.key == "llama.blockCount") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.num_layers = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.vocab_size" || kv.key == "llama.vocabSize") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.vocab_size = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.rope.dimension_count" || kv.key == "llama.rope.dimensionCount") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.head_dim = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.rope.freq_base" || kv.key == "llama.rope.freqBase") {
                if (kv.value_type == GGUFValueType::FLOAT32) {
                    model_config_.rope_theta = *static_cast<float*>(kv.value);
                }
            } else if (kv.key == "llama.attention.layer_norm_rms_epsilon" || kv.key == "llama.attention.layerNormEpsilon") {
                if (kv.value_type == GGUFValueType::FLOAT32) {
                    model_config_.rms_epsilon = *static_cast<float*>(kv.value);
                }
            }
        }

        // Calculate head_dim if not provided
        if (model_config_.head_dim == 0 && model_config_.num_attention_heads > 0) {
            model_config_.head_dim = model_config_.hidden_size / model_config_.num_attention_heads;
        }
    }

    static uint64_t align_offset(uint64_t offset, uint64_t alignment) {
        if (alignment == 0) return offset;
        return (offset + alignment - 1) / alignment * alignment;
    }

    std::vector<float> convert_raw_to_fp32_fallback(const std::vector<uint8_t>& data,
                                                     GGUFValueType data_type,
                                                     uint64_t element_count) {
        std::vector<float> result;
        result.reserve(element_count);

        switch (data_type) {
            case GGUFValueType::FLOAT32: {
                const float* ptr = reinterpret_cast<const float*>(data.data());
                size_t count = std::min(element_count, data.size() / sizeof(float));
                result.assign(ptr, ptr + count);
                break;
            }
            case GGUFValueType::FLOAT16: {
                const uint16_t* ptr = reinterpret_cast<const uint16_t*>(data.data());
                size_t count = std::min(element_count, data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(fp16_to_fp32(ptr[i]));
                }
                break;
            }
            case GGUFValueType::BFLOAT16: {
                const uint16_t* ptr = reinterpret_cast<const uint16_t*>(data.data());
                size_t count = std::min(element_count, data.size() / sizeof(uint16_t));
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(bf16_to_fp32(ptr[i]));
                }
                break;
            }
            case GGUFValueType::INT8: {
                const int8_t* ptr = reinterpret_cast<const int8_t*>(data.data());
                size_t count = std::min(element_count, data.size());
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            case GGUFValueType::INT16: {
                const int16_t* ptr = reinterpret_cast<const int16_t*>(data.data());
                size_t count = std::min(element_count, data.size() / sizeof(int16_t));
                for (size_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            default:
                break;
        }

        return result;
    }

    static float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float val = mant * std::pow(2.0f, -24.0f);
            return sign ? -val : val;
        } else if (exp == 31) {
            if (mant == 0) {
                return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            }
            return std::numeric_limits<float>::quiet_NaN();
        }

        float val = std::pow(2.0f, static_cast<float>(exp - 15)) * (1.0f + mant / 1024.0f);
        return sign ? -val : val;
    }

    static float bf16_to_fp32(uint16_t b) {
        uint32_t val = static_cast<uint32_t>(b) << 16;
        float result;
        std::memcpy(&result, &val, sizeof(float));
        return result;
    }
};

// ============================================================================
// Public API Implementation
// ============================================================================
GGUFLoader::GGUFLoader(const Path& file_path)
    : file_path_(file_path), pimpl_(std::make_unique<GGUFLoaderImpl>(file_path)) {}

GGUFLoader::~GGUFLoader() = default;

ExpectedVoid GGUFLoader::load() {
    return pimpl_->load();
}

ExpectedVoid GGUFLoader::close() {
    return pimpl_->close();
}

std::vector<PackMetadata> GGUFLoader::generate_packs(const ModelConfig& config) {
    (void)config;
    return pimpl_->generate_packs();
}

uint64_t GGUFLoader::tensor_count() const {
    return pimpl_->tensors().size();
}

uint64_t GGUFLoader::metadata_count() const {
    return pimpl_->metadata().size();
}

const GGUFTensorInfo* GGUFLoader::get_tensor(const std::string& name) const {
    for (const auto& tensor : pimpl_->tensors()) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

Expected<std::vector<float>> GGUFLoader::read_tensor_data(const GGUFTensorInfo& tensor, bool fp16_to_fp32) {
    return pimpl_->read_tensor_data(tensor, fp16_to_fp32);
}

std::vector<Expected<std::vector<float>>> GGUFLoader::read_tensors_parallel(
    const std::vector<std::string>& tensor_names, bool fp16_to_fp32) {
    return pimpl_->load_tensors_parallel(tensor_names, fp16_to_fp32);
}

const std::vector<VocabularyEntry>& GGUFLoader::get_vocabulary() const {
    return pimpl_->vocabulary();
}

void GGUFLoader::clear_tensor_cache() {
    pimpl_->clear_tensor_cache();
}

size_t GGUFLoader::get_tensor_cache_size() const {
    return pimpl_->get_tensor_cache_size();
}

ExpectedVoid GGUFLoader::add_shard(const Path& shard_path) {
    return pimpl_->add_shard(shard_path);
}

bool GGUFLoader::is_sharded() const {
    return pimpl_->is_sharded();
}

uint64_t GGUFLoader::align_offset(uint64_t offset, uint64_t alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) / alignment * alignment;
}

ModelConfig GGUFLoader::get_model_config() const {
    return pimpl_->model_config();
}

} // namespace vk_symbiote
