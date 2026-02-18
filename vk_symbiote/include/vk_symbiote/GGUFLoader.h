#pragma once

#include "Common.h"
#include "NomadPack.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <shared_mutex>
#include <unordered_map>
#include <list>

namespace vk_symbiote {

// Forward declaration for PIMPL idiom
class GGUFLoaderImpl;
class WorkStealingThreadPool;

enum class GGUFValueType : uint32 {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    Q4_0 = 13,
    Q4_1 = 14,
    Q5_0 = 15,
    Q5_1 = 16,
    Q8_0 = 17,
    Q8_1 = 18,
    Q2_K = 19,
    Q3_K = 20,
    Q4_K = 21,
    Q5_K = 22,
    Q6_K = 23,
    Q8_K = 24,
    IQ1_S = 25,
    IQ1_M = 26,
    IQ2_XXS = 27,
    IQ2_XS = 28,
    IQ3_XXS = 29,
    IQ4_XS = 30,
    IQ4_NL = 31,
    IQ5_S = 32,
    IQ5_M = 33,
    IQ6_N = 34,
    IQ6_1 = 35,
    IQ2_S = 36,
    IQ3_S = 37,
    IQ4_1 = 38,
    IQ3_1 = 39,
    IQ2_1 = 40,
    IQ1_1 = 41,
    IQ4_NL_2 = 42,
    IQ3_L = 43,
    IQ3_XS = 44,
    IQ3_M = 45,
    IQ4_XS_2 = 46,
    BFLOAT16 = 47,
    FLOAT16 = 48,
    IQ2_M = 49,
    IQ3_XS_2 = 50,
    IQ3_L_2 = 51,
    IQ3_M_2 = 52,
    IQ4_XS_3 = 53
};

struct GGUFHeader {
    uint32 magic = 0;
    uint32 version = 0;
    uint64 tensor_count = 0;
    uint64 metadata_kv_count = 0;
};

struct GGUFTensorInfo {
    std::string name;
    uint32 n_dimensions = 0;
    std::vector<uint64> dimensions;
    GGUFValueType data_type;
    uint64 offset;
    uint64 data_offset = 0;  // Absolute offset in file
    uint64 data_size = 0;    // Size of raw data in bytes
    bool is_compressed = false;
    uint32 compression_codec = 0;  // 0=none, 1=blosc2, 2=zfp, 3=hybrid
};

struct GGUFMetadataKV {
    std::string key;
    GGUFValueType value_type;
    std::vector<uint8_t> raw_value;
    
    // Parsed value accessors
    std::string get_string() const;
    uint32_t get_uint32() const;
    int32_t get_int32() const;
    float get_float32() const;
    bool get_bool() const;
    std::vector<uint8_t> get_array_data() const;
    GGUFValueType get_array_element_type() const;
};

// Vocabulary entry for tokenizer integration
struct VocabularyEntry {
    std::string token;
    std::vector<uint8_t> bytes;
    float score = 0.0f;
    uint32_t token_type = 0;
    bool is_special = false;
};

// Tokenizer configuration
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
    std::vector<std::pair<std::string, uint32_t>> added_tokens;
};

// LRU Cache configuration and entry
struct TensorCacheConfig {
    size_t max_entries = 256;
    size_t max_memory_bytes = 4ULL * 1024 * 1024 * 1024;  // 4GB default
    uint64_t ttl_seconds = 300;  // 5 minutes default
    bool enable_compression = true;
};

struct TensorCacheEntry {
    std::vector<float> data;
    uint64_t timestamp;
    uint64_t access_count;
    size_t memory_cost;
    GGUFTensorInfo tensor_info;
};

class TensorLRUCache {
public:
    explicit TensorLRUCache(const TensorCacheConfig& config = TensorCacheConfig{});
    
    // Get tensor from cache
    bool get(const std::string& tensor_name, std::vector<float>& out_data);
    
    // Put tensor in cache
    void put(const std::string& tensor_name, const std::vector<float>& data, 
             const GGUFTensorInfo& info);
    
    // Check if tensor is cached
    bool contains(const std::string& tensor_name) const;
    
    // Clear all cache entries
    void clear();
    
    // Get cache statistics
    struct Stats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t current_entries = 0;
        size_t current_memory_bytes = 0;
        double hit_rate() const { 
            auto total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };
    Stats get_stats() const;
    
    // Touch entry to mark as recently used
    void touch(const std::string& tensor_name);
    
    // Remove expired entries
    size_t remove_expired();
    
private:
    TensorCacheConfig config_;
    mutable std::shared_mutex mutex_;
    
    struct CacheNode {
        std::string key;
        TensorCacheEntry entry;
    };
    
    std::list<CacheNode> cache_list_;
    std::unordered_map<std::string, typename std::list<CacheNode>::iterator> cache_map_;
    
    size_t current_memory_ = 0;
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
    mutable size_t evictions_ = 0;
    
    void evict_oldest();
    bool should_evict(size_t new_memory) const;
};

// Streaming tensor reader for async multi-thread loading
class StreamingTensorReader {
public:
    struct ReadRequest {
        uint64_t file_offset;
        uint64_t size;
        std::string tensor_name;
        GGUFValueType data_type;
        std::vector<uint64_t> dimensions;
        bool decompress = true;
    };
    
    struct ReadResult {
        std::string tensor_name;
        std::vector<float> data;
        bool success = false;
        std::string error_msg;
        uint64_t read_time_us = 0;
    };
    
    explicit StreamingTensorReader(const Path& file_path, 
                                   WorkStealingThreadPool* thread_pool = nullptr);
    ~StreamingTensorReader();
    
    // Read single tensor synchronously
    Expected<std::vector<float>> read_tensor(const GGUFTensorInfo& tensor_info);
    
    // Read multiple tensors in parallel using thread pool
    std::vector<Expected<std::vector<float>>> read_tensors_parallel(
        const std::vector<GGUFTensorInfo>& tensors);
    
    // Read tensor with hybrid decompression
    Expected<std::vector<float>> read_tensor_hybrid(const GGUFTensorInfo& tensor_info);
    
    // Async read with callback
    void read_tensor_async(const ReadRequest& request,
                          std::function<void(ReadResult)> callback);
    
    // Batch async read
    void read_tensors_async_batch(const std::vector<ReadRequest>& requests,
                                  std::function<void(std::vector<ReadResult>)> batch_callback);
    
    // Get file handle
    std::ifstream& get_file_handle() { return file_; }
    
private:
    Path file_path_;
    std::ifstream file_;
    WorkStealingThreadPool* thread_pool_;
    std::mutex file_mutex_;
    
    Expected<std::vector<float>> decompress_data(const std::vector<uint8_t>& compressed_data,
                                                  const GGUFTensorInfo& info);
    Expected<std::vector<float>> dequantize_data(const std::vector<uint8_t>& raw_data,
                                                  const GGUFTensorInfo& info);
};

class GGUFLoader {
public:
    explicit GGUFLoader(const Path& file_path);
    ~GGUFLoader();

    ExpectedVoid load();
    ExpectedVoid close();

    ModelConfig get_model_config() const;

    std::vector<PackMetadata> generate_packs(const ModelConfig& config);

    uint64 tensor_count() const;
    uint64 metadata_count() const;

    const GGUFTensorInfo* get_tensor(const std::string& name) const;
    Expected<std::vector<float>> read_tensor_data(const GGUFTensorInfo& tensor, 
                                                   bool fp16_to_fp32 = true);
    
    // Read tensor with cache integration
    Expected<std::vector<float>> read_tensor_cached(const std::string& tensor_name,
                                                     bool fp16_to_fp32 = true);
    
    // Parallel tensor loading using thread pool
    std::vector<Expected<std::vector<float>>> read_tensors_parallel(
        const std::vector<std::string>& tensor_names, bool fp16_to_fp32 = true);
    
    // Parallel streaming with multi-thread chunk processing
    std::vector<Expected<std::vector<float>>> stream_tensors_parallel(
        const std::vector<std::string>& tensor_names, 
        uint32_t num_chunks = 4);
    
    // Vocabulary access for tokenizer
    const std::vector<VocabularyEntry>& get_vocabulary() const;
    const TokenizerConfig& get_tokenizer_config() const;
    
    // Cache management
    void clear_tensor_cache();
    size_t get_tensor_cache_size() const;
    TensorLRUCache::Stats get_cache_stats() const;
    void configure_cache(const TensorCacheConfig& config);
    
    // Multi-file shard support
    ExpectedVoid add_shard(const Path& shard_path);
    bool is_sharded() const;
    
    // Get all tensor names
    std::vector<std::string> get_tensor_names() const;
    
    // Metadata accessors
    std::vector<std::string> get_metadata_keys() const;
    const GGUFMetadataKV* get_metadata(const std::string& key) const;

    static uint64 align_offset(uint64 offset, uint64 alignment);
    
    // Version info
    uint32_t get_version() const;
    bool is_valid() const;

private:
    Path file_path_;
    std::unique_ptr<GGUFLoaderImpl> pimpl_;
    std::unique_ptr<TensorLRUCache> tensor_cache_;
    std::unique_ptr<StreamingTensorReader> tensor_reader_;
    std::unique_ptr<WorkStealingThreadPool> thread_pool_;
    
    bool loaded_ = false;
    uint32_t version_ = 0;
};

} // namespace vk_symbiote
