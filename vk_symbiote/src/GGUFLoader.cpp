#include "GGUFLoader.h"
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

namespace vk_symbiote {

// ============================================================================
// Thread Pool for Parallel Tensor Reading
// ============================================================================
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : stop_(false), active_tasks_(0) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
            ++active_tasks_;
        }
        
        condition_.notify_one();
        return result;
    }
    
    size_t active_tasks() const { return active_tasks_.load(); }
    
    void wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        wait_condition_.wait(lock, [this] { 
            return tasks_.empty() && active_tasks_ == 0; 
        });
    }
    
    size_t num_threads() const { return workers_.size(); }
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable wait_condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_tasks_;
    
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                
                if (stop_ && tasks_.empty()) {
                    return;
                }
                
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            
            task();
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                --active_tasks_;
            }
            wait_condition_.notify_all();
        }
    }
};

// ============================================================================
// LRU Cache for Tensor Data with Memory Pressure Awareness
// ============================================================================
template<typename K, typename V>
class LRUCache {
public:
    explicit LRUCache(size_t max_size = 100, size_t max_memory_bytes = 0) 
        : max_size_(max_size), max_memory_bytes_(max_memory_bytes), current_memory_(0) {}
    
    bool get(const K& key, V& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return false;
        }
        // Move to front (most recently used)
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        value = it->second->second;
        return true;
    }
    
    void put(const K& key, const V& value, size_t memory_cost = 0) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Update existing entry
            current_memory_ -= it->second->memory_cost;
            it->second->second = value;
            it->second->memory_cost = memory_cost;
            current_memory_ += memory_cost;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }
        
        // Evict oldest if at capacity or memory limit
        while (cache_list_.size() >= max_size_ || 
               (max_memory_bytes_ > 0 && current_memory_ + memory_cost > max_memory_bytes_)) {
            if (cache_list_.empty()) break;
            
            auto last = cache_list_.end();
            --last;
            current_memory_ -= last->memory_cost;
            cache_map_.erase(last->key);
            cache_list_.pop_back();
        }
        
        // Insert new entry at front
        CacheEntry entry;
        entry.key = key;
        entry.second = value;
        entry.memory_cost = memory_cost;
        
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
    
    float hit_rate() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        uint64_t total = hits_ + misses_;
        return total > 0 ? static_cast<float>(hits_) / static_cast<float>(total) : 0.0f;
    }
    
    void record_hit() { std::unique_lock<std::shared_mutex> lock(mutex_); ++hits_; }
    void record_miss() { std::unique_lock<std::shared_mutex> lock(mutex_); ++misses_; }
    
private:
    struct CacheEntry {
        K key;
        V second;
        size_t memory_cost = 0;
    };
    
    size_t max_size_;
    size_t max_memory_bytes_;
    size_t current_memory_;
    mutable std::list<CacheEntry> cache_list_;
    mutable std::unordered_map<K, typename std::list<CacheEntry>::iterator> cache_map_;
    mutable std::shared_mutex mutex_;
    mutable uint64_t hits_ = 0;
    mutable uint64_t misses_ = 0;
};

// ============================================================================
// Enhanced Vocabulary Parser with Full Tokenizer.ggml Support
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
    };
    
    std::pair<std::vector<VocabularyEntry>, TokenizerConfig> parse_from_metadata(
        std::istream& file,
        const std::vector<GGUFMetadataKV>& metadata) {
        
        std::vector<VocabularyEntry> vocab;
        TokenizerConfig config;
        
        // First pass: parse tokenizer configuration
        for (const auto& kv : metadata) {
            parse_tokenizer_config(kv, config);
        }
        
        // Second pass: parse vocabulary tokens
        // Look for tokenizer.ggml.tokens (array of strings)
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.tokens" && kv.value_type == GGUFValueType::ARRAY) {
                vocab = parse_token_array(file, kv.value);
            }
        }
        
        // Third pass: parse scores if available
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.scores" && kv.value_type == GGUFValueType::ARRAY) {
                parse_scores(kv.value, vocab);
            }
        }
        
        // Fourth pass: parse token types if available
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.token_type" && kv.value_type == GGUFValueType::ARRAY) {
                parse_token_types(kv.value, vocab);
            }
        }
        
        // Fifth pass: parse merges for BPE tokenizers
        for (const auto& kv : metadata) {
            if (kv.key == "tokenizer.ggml.merges" && kv.value_type == GGUFValueType::ARRAY) {
                // Store merges for later use in tokenizer
                // Not storing in vocab entries directly
            }
        }
        
        return {vocab, config};
    }
    
private:
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
        }
    }
    
    std::vector<VocabularyEntry> parse_token_array(std::istream& file, void* value) {
        std::vector<VocabularyEntry> result;
        
        if (!value) return result;
        
        // The value points to array metadata stored in the GGUF file
        // Format: [element_type: uint32][element_count: uint64][data...]
        auto* array_data = static_cast<std::vector<uint8_t>*>(value);
        
        if (array_data->size() < 12) return result; // Need at least header
        
        const uint8_t* data = array_data->data();
        GGUFValueType elem_type = static_cast<GGUFValueType>(*reinterpret_cast<const uint32_t*>(data));
        uint64_t count = *reinterpret_cast<const uint64_t*>(data + 4);
        
        if (elem_type != GGUFValueType::STRING) {
            // Only string arrays are supported for tokens
            return result;
        }
        
        result.reserve(count);
        size_t offset = 12; // Skip header
        
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
};

// Static tensor type size lookup table
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
            return 18; // 4 bits per element, 32 elements per block + metadata
        case GGUFValueType::Q5_0:
        case GGUFValueType::Q5_1:
            return 22; // 5 bits per element
        case GGUFValueType::Q8_0:
        case GGUFValueType::Q8_1:
            return 34; // 8 bits per element
        default:
            return 4; // Default to float32
    }
}

// Calculate total tensor size in bytes
static uint64_t calculate_tensor_size(const GGUFTensorInfo& tensor) {
    uint64_t element_count = 1;
    for (uint64_t dim : tensor.dimensions) {
        element_count *= dim;
    }
    
    uint64_t type_size = get_type_size(tensor.data_type);
    
    // Handle quantized types with block structure
    if (tensor.data_type >= GGUFValueType::Q4_0 && tensor.data_type <= GGUFValueType::Q8_1) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        return num_blocks * type_size;
    }
    
    return element_count * type_size;
}

// Streaming tensor reader for on-demand loading with multi-thread support
class StreamingTensorReader {
public:
    explicit StreamingTensorReader(const std::filesystem::path& path) 
        : file_path_(path), file_stream_(), mutex_() {}
    
    ~StreamingTensorReader() {
        close();
    }
    
    bool open() {
        std::lock_guard<std::mutex> lock(mutex_);
        file_stream_.open(file_path_, std::ios::binary | std::ios::in);
        return file_stream_.is_open();
    }
    
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
    }
    
    // Read a chunk of tensor data at specified offset (thread-safe)
    bool read_chunk(uint64_t offset, void* buffer, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_stream_.is_open()) return false;
        
        file_stream_.seekg(static_cast<std::streamoff>(offset));
        if (!file_stream_.good()) return false;
        
        file_stream_.read(static_cast<char*>(buffer), static_cast<std::streamsize>(size));
        return file_stream_.good() || file_stream_.gcount() == static_cast<std::streamsize>(size);
    }
    
    // Async read using thread pool
    std::future<bool> read_chunk_async(ThreadPool& pool, uint64_t offset, void* buffer, size_t size) {
        return pool.enqueue([this, offset, buffer, size]() {
            return read_chunk(offset, buffer, size);
        });
    }
    
    // Read tensor data in parallel chunks
    bool read_parallel(ThreadPool& pool, uint64_t offset, void* buffer, size_t total_size, 
                       size_t chunk_size = 4 * 1024 * 1024) { // 4MB chunks
        if (total_size <= chunk_size) {
            return read_chunk(offset, buffer, total_size);
        }
        
        char* byte_buffer = static_cast<char*>(buffer);
        size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;
        std::vector<std::future<bool>> futures;
        futures.reserve(num_chunks);
        
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t current_offset = i * chunk_size;
            size_t current_size = std::min(chunk_size, total_size - current_offset);
            
            futures.push_back(
                pool.enqueue([this, offset, byte_buffer, current_offset, current_size]() {
                    return read_chunk(offset + current_offset, byte_buffer + current_offset, current_size);
                })
            );
        }
        
        // Wait for all chunks
        bool success = true;
        for (auto& f : futures) {
            success = success && f.get();
        }
        
        return success;
    }
    
private:
    std::filesystem::path file_path_;
    std::ifstream file_stream_;
    std::mutex mutex_;
};

// Tensor pack mapping - maps GGUF tensors to NomadPack structures
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
    };
    
    std::vector<PackMapping> create_mappings(
        const std::vector<GGUFTensorInfo>& tensors,
        const ModelConfig& config) {
        
        std::vector<PackMapping> mappings;
        mappings.reserve(tensors.size());
        uint64_t next_pack_id = 0;
        
        // Group tensors by layer for efficient access patterns
        std::unordered_map<uint32_t, std::vector<const GGUFTensorInfo*>> layer_tensors;
        
        for (const auto& tensor : tensors) {
            uint32_t layer_idx = extract_layer_index(tensor.name);
            layer_tensors[layer_idx].push_back(&tensor);
        }
        
        // Create pack mappings for each layer
        for (const auto& [layer_idx, tensor_ptrs] : layer_tensors) {
            for (const auto* tensor : tensor_ptrs) {
                PackMapping mapping;
                mapping.pack_id = next_pack_id++;
                mapping.tensor_name = tensor->name;
                mapping.file_offset = tensor->offset;
                mapping.data_type = tensor->data_type;
                mapping.dimensions = tensor->dimensions;
                mapping.layer_idx = layer_idx;
                mapping.pack_type = infer_pack_type(tensor->name);
                
                // Calculate sizes
                mapping.decompressed_size = calculate_tensor_size(*tensor);
                mapping.compressed_size = mapping.decompressed_size; // GGUF stores raw, we'll compress in NomadPack
                
                mappings.push_back(mapping);
            }
        }
        
        return mappings;
    }
    
private:
    uint32_t extract_layer_index(const std::string& tensor_name) {
        // Parse layer index from tensor name (e.g., "layers.5.attention.wq.weight" -> 5)
        size_t pos = tensor_name.find("layers.");
        if (pos != std::string::npos) {
            size_t start = pos + 7; // length of "layers."
            size_t end = tensor_name.find('.', start);
            if (end != std::string::npos) {
                try {
                    return std::stoul(tensor_name.substr(start, end - start));
                } catch (...) {
                    return 0;
                }
            }
        }
        return 0;
    }
    
    PackType infer_pack_type(const std::string& tensor_name) {
        if (tensor_name.find("attention.wq") != std::string::npos) return PackType::ATTENTION_Q;
        if (tensor_name.find("attention.wk") != std::string::npos) return PackType::ATTENTION_K;
        if (tensor_name.find("attention.wv") != std::string::npos) return PackType::ATTENTION_V;
        if (tensor_name.find("attention.wo") != std::string::npos) return PackType::ATTENTION_O;
        if (tensor_name.find("feed_forward.w1") != std::string::npos) return PackType::FEED_FORWARD_UP;
        if (tensor_name.find("feed_forward.w2") != std::string::npos) return PackType::FEED_FORWARD_DOWN;
        if (tensor_name.find("feed_forward.w3") != std::string::npos) return PackType::FEED_FORWARD_GATE;
        if (tensor_name.find("attention_norm") != std::string::npos) return PackType::NORM_GAMMA;
        if (tensor_name.find("ffn_norm") != std::string::npos) return PackType::NORM_GAMMA;
        if (tensor_name.find("rope") != std::string::npos) return PackType::ROPE;
        if (tensor_name.find("embed") != std::string::npos || tensor_name.find("token_embd") != std::string::npos) 
            return PackType::EMBEDDING;
        if (tensor_name.find("output") != std::string::npos || tensor_name.find("output_norm") != std::string::npos)
            return PackType::HEAD;
        return PackType::UNKNOWN;
    }
};

// ============================================================================
// Hybrid Decompression Engine
// ============================================================================
class HybridDecompressor {
public:
    HybridDecompressor() {
        compression::Blosc2Compression::initialize();
    }
    
    std::vector<float> decompress(const std::vector<uint8_t>& compressed_data, 
                                   uint64_t expected_elements,
                                   GGUFValueType data_type) {
        std::vector<float> result;
        result.reserve(expected_elements);
        
        // Check for compression header
        if (compressed_data.size() >= 16) {
            // Check for Blosc2 header (0x0201)
            if (compressed_data[0] == 0x02 && compressed_data[1] == 0x01) {
                return decompress_blosc2(compressed_data, expected_elements);
            }
            
            // Check for ZFP header
            if (compressed_data[0] == 'Z' && compressed_data[1] == 'F' && compressed_data[2] == 'P') {
                return decompress_zfp(compressed_data, expected_elements);
            }
            
            // Check for custom hybrid compression
            if (compressed_data[0] == 'V' && compressed_data[1] == 'K' && compressed_data[2] == 'S') {
                return decompress_hybrid(compressed_data, expected_elements, data_type);
            }
        }
        
        // Raw data - convert based on type
        return convert_raw_to_fp32(compressed_data, data_type, expected_elements);
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
        // ZFP decompression would go here
        // For now, return empty (fallback to raw)
        (void)compressed_data;
        (void)expected_elements;
        return {};
    }
    
    std::vector<float> decompress_hybrid(const std::vector<uint8_t>& compressed_data,
                                          uint64_t expected_elements,
                                          GGUFValueType data_type) {
        // Hybrid decompression: first decompress with Blosc2, then apply quantization if needed
        auto decompressed = decompress_blosc2(compressed_data, expected_elements);
        
        if (decompressed.empty() && is_quantized_type(data_type)) {
            // Try direct quantization decompression
            return decompress_quantized(compressed_data, data_type, expected_elements);
        }
        
        return decompressed;
    }
    
    std::vector<float> decompress_quantized(const std::vector<uint8_t>& data,
                                            GGUFValueType data_type,
                                            uint64_t expected_elements) {
        std::vector<float> result;
        result.reserve(expected_elements);
        
        switch (data_type) {
            case GGUFValueType::Q8_0:
                result = decompress_q8_0(data, expected_elements);
                break;
            case GGUFValueType::Q4_0:
                result = decompress_q4_0(data, expected_elements);
                break;
            case GGUFValueType::Q5_0:
            case GGUFValueType::Q5_1:
                // Q5 decompression
                break;
            default:
                break;
        }
        
        return result;
    }
    
    std::vector<float> decompress_q8_0(const std::vector<uint8_t>& data, uint64_t element_count) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);
        
        const uint8_t* ptr = data.data();
        size_t offset = 0;
        
        for (uint64_t block = 0; block < num_blocks && offset + block_size + 4 <= data.size(); ++block) {
            float scale = *reinterpret_cast<const float*>(ptr + offset);
            offset += 4;
            
            for (uint32_t i = 0; i < block_size && result.size() < element_count && offset < data.size(); ++i) {
                int8_t quantized = static_cast<int8_t>(ptr[offset++]);
                result.push_back(scale * static_cast<float>(quantized));
            }
        }
        
        return result;
    }
    
    std::vector<float> decompress_q4_0(const std::vector<uint8_t>& data, uint64_t element_count) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);
        
        const uint8_t* ptr = data.data();
        size_t offset = 0;
        
        for (uint64_t block = 0; block < num_blocks && offset + block_size / 2 + 4 <= data.size(); ++block) {
            float scale = *reinterpret_cast<const float*>(ptr + offset);
            offset += 4;
            
            for (uint32_t i = 0; i < block_size / 2 && result.size() < element_count; ++i) {
                if (offset >= data.size()) break;
                
                uint8_t packed = ptr[offset++];
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
            if (mant == 0) {
                return sign ? -0.0f : 0.0f;
            }
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

// Full GGUFLoader implementation with streaming and multi-thread support
class GGUFLoaderImpl {
public:
    explicit GGUFLoaderImpl(const Path& file_path) 
        : file_path_(file_path), data_offset_(0), thread_pool_(std::thread::hardware_concurrency()), 
          tensor_cache_(50, 2ULL * 1024 * 1024 * 1024), // 50 entries, 2GB max memory
          hybrid_decompressor_(std::make_unique<HybridDecompressor>()),
          is_sharded_(false) {
        // Initialize Blosc2 compression backend
        compression::Blosc2Compression::initialize();
    }
    
    ExpectedVoid load() {
        // Open file
        file_.open(file_path_.c_str(), std::ios::binary);
        if (!file_.is_open()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        // Read header
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
        
        // Parse model configuration from metadata
        parse_model_config();
        
        // Create tensor pack mappings
        TensorPackMapper mapper;
        tensor_mappings_ = mapper.create_mappings(tensors_, model_config_);
        
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
    
    // On-demand tensor data loading with LRU cache and hybrid decompression
    Expected<std::vector<float>> load_tensor_data(const std::string& tensor_name, bool convert_fp16 = true) {
        // Check cache first
        std::vector<float> cached_result;
        if (tensor_cache_.get(tensor_name, cached_result)) {
            tensor_cache_.record_hit();
            return Expected<std::vector<float>>(std::move(cached_result));
        }
        tensor_cache_.record_miss();
        
        const GGUFTensorInfo* tensor = find_tensor(tensor_name);
        if (!tensor) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        uint64_t tensor_size = calculate_tensor_size(*tensor);
        uint64_t element_count = tensor_size / get_type_size(tensor->data_type);
        
        // For FP16->FP32 conversion
        if (tensor->data_type == GGUFValueType::FLOAT16 && convert_fp16) {
            element_count = tensor_size / 2; // FP16 is 2 bytes
        }
        
        // Read tensor data in parallel chunks
        std::vector<uint8_t> raw_data(tensor_size);
        bool read_success = stream_reader_->read_parallel(thread_pool_, tensor->offset, 
                                                          raw_data.data(), tensor_size);
        
        if (!read_success) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        // Apply hybrid decompression
        std::vector<float> result = hybrid_decompressor_->decompress(raw_data, element_count, tensor->data_type);
        
        if (result.empty()) {
            // Fallback to direct type conversion
            result = convert_raw_to_fp32(raw_data, tensor->data_type, element_count);
        }
        
        // Store in cache with memory cost tracking
        if (!result.empty()) {
            size_t memory_cost = result.size() * sizeof(float);
            tensor_cache_.put(tensor_name, result, memory_cost);
        }
        
        return Expected<std::vector<float>>(std::move(result));
    }
    
    // Read tensor data with explicit tensor info (public API)
    Expected<std::vector<float>> read_tensor_data(const GGUFTensorInfo& tensor, bool convert_fp16 = true) {
        return load_tensor_data(tensor.name, convert_fp16);
    }
    
    // Parallel load multiple tensors using thread pool
    std::vector<Expected<std::vector<float>>> load_tensors_parallel(
        const std::vector<std::string>& tensor_names, bool convert_fp16 = true) {
        
        std::vector<Expected<std::vector<float>>> results(tensor_names.size());
        std::vector<std::future<void>> futures;
        
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            futures.push_back(thread_pool_.enqueue([this, &tensor_names, &results, i, convert_fp16]() {
                results[i] = load_tensor_data(tensor_names[i], convert_fp16);
            }));
        }
        
        // Wait for all to complete
        for (auto& f : futures) {
            f.wait();
        }
        
        return results;
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
            pack.file_offset = mapping.file_offset;
            pack.compressed_size = mapping.compressed_size;
            pack.decompressed_size = mapping.decompressed_size;
            pack.tensor_name = mapping.tensor_name;
            
            // Set base priority based on layer and type
            float layer_factor = 1.0f - (static_cast<float>(mapping.layer_idx) / model_config_.num_layers);
            float type_factor = 0.8f;
            
            switch (mapping.pack_type) {
                case PackType::ATTENTION_Q:
                case PackType::ATTENTION_K:
                case PackType::ATTENTION_V:
                    type_factor = 1.0f;
                    break;
                case PackType::ATTENTION_O:
                    type_factor = 0.9f;
                    break;
                case PackType::FEED_FORWARD_UP:
                case PackType::FEED_FORWARD_GATE:
                case PackType::FEED_FORWARD_DOWN:
                    type_factor = 0.7f;
                    break;
                default:
                    type_factor = 0.5f;
                    break;
            }
            
            pack.base_priority = layer_factor * type_factor;
            packs.push_back(pack);
        }
        
        return packs;
    }
    
    // Multi-file shard support
    ExpectedVoid add_shard(const Path& shard_path) {
        // Store shard path for later loading
        shard_paths_.push_back(shard_path);
        is_sharded_ = true;
        return make_expected_success();
    }
    
    bool is_sharded() const { return is_sharded_; }
    
    // Cache management
    void clear_tensor_cache() {
        tensor_cache_.clear();
    }
    
    size_t get_tensor_cache_size() const {
        return tensor_cache_.size();
    }
    
    float get_tensor_cache_hit_rate() const {
        return tensor_cache_.hit_rate();
    }
    
    size_t get_tensor_cache_memory() const {
        return tensor_cache_.memory_usage();
    }
    
    const GGUFHeader& header() const { return header_; }
    const ModelConfig& model_config() const { return model_config_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    const std::vector<GGUFMetadataKV>& metadata() const { return metadata_; }
    const std::vector<VocabularyEntry>& vocabulary() const { return vocabulary_; }
    
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
    
    // Thread pool for parallel tensor reads
    ThreadPool thread_pool_;
    
    // LRU cache for tensor data (tensor_name -> float data)
    LRUCache<std::string, std::vector<float>> tensor_cache_;
    
    // Hybrid decompression engine
    std::unique_ptr<HybridDecompressor> hybrid_decompressor_;
    
    // Vocabulary entries parsed from metadata
    std::vector<VocabularyEntry> vocabulary_;
    VocabularyParser::TokenizerConfig tokenizer_config_;
    
    // Multi-file shard support
    std::vector<Path> shard_paths_;
    bool is_sharded_;
    
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
            
            // Track maximum offset for data section
            uint64_t tensor_end = tensor.offset + calculate_tensor_size(tensor);
            current_offset = std::max(current_offset, tensor_end);
        }
        
        // Data section starts after aligned tensor info
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
                // Arrays are complex - read element type and count
                GGUFValueType elem_type;
                uint64_t count;
                file_.read(reinterpret_cast<char*>(&elem_type), 4);
                file_.read(reinterpret_cast<char*>(&count), 8);
                
                // Store as raw bytes for later parsing
                std::vector<uint8_t>* val = new std::vector<uint8_t>;
                val->resize(12 + count * get_type_size(elem_type)); // Header + data
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
            case GGUFValueType::UINT8:
                delete static_cast<uint8_t*>(value);
                break;
            case GGUFValueType::INT8:
                delete static_cast<int8_t*>(value);
                break;
            case GGUFValueType::UINT16:
                delete static_cast<uint16_t*>(value);
                break;
            case GGUFValueType::INT16:
                delete static_cast<int16_t*>(value);
                break;
            case GGUFValueType::UINT32:
                delete static_cast<uint32_t*>(value);
                break;
            case GGUFValueType::INT32:
                delete static_cast<int32_t*>(value);
                break;
            case GGUFValueType::FLOAT32:
                delete static_cast<float*>(value);
                break;
            case GGUFValueType::UINT64:
                delete static_cast<uint64_t*>(value);
                break;
            case GGUFValueType::INT64:
                delete static_cast<int64_t*>(value);
                break;
            case GGUFValueType::FLOAT64:
                delete static_cast<double*>(value);
                break;
            case GGUFValueType::BOOL:
                delete static_cast<bool*>(value);
                break;
            case GGUFValueType::STRING:
                delete static_cast<std::string*>(value);
                break;
            case GGUFValueType::ARRAY:
                delete static_cast<std::vector<uint8_t>*>(value);
                break;
            default:
                delete[] static_cast<char*>(value);
                break;
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
    
    const GGUFTensorInfo* find_tensor(const std::string& name) const {
        for (const auto& tensor : tensors_) {
            if (tensor.name == name) {
                return &tensor;
            }
        }
        return nullptr;
    }
    
    static uint64_t align_offset(uint64_t offset, uint64_t alignment) {
        return (offset + alignment - 1) / alignment * alignment;
    }
    
    std::vector<float> convert_raw_to_fp32(const std::vector<uint8_t>& data,
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
            case GGUFValueType::INT32: {
                const int32_t* ptr = reinterpret_cast<const int32_t*>(data.data());
                size_t count = std::min(element_count, data.size() / sizeof(int32_t));
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
            if (mant == 0) {
                return sign ? -0.0f : 0.0f;
            }
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

// Public API implementation using PIMPL pattern
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
    (void)config; // Config already parsed during load
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
