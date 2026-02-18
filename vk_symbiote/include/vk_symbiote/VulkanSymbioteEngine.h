#pragma once

#include "Common.h"
#include "NomadPack.h"
#include "VitalityOracle.h"
#include "GGUFLoader.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace vk_symbiote {

class ShaderRuntime;
class Tokenizer;
class KVCacheManager;
class PowerManager;

// Device capabilities for shader tuning
struct DeviceCapabilities {
    uint32_t max_compute_workgroup_invocations = 256;
    uint32_t max_compute_workgroup_size[3] = {256, 1, 1};
    uint32_t subgroup_size = 32;
    uint32_t max_push_constant_size = 128;
    uint32_t shared_memory_size = 16384;
    bool supports_fp16 = false;
    bool supports_int8 = false;
    bool supports_cooperative_matrix = false;
    uint32_t cooperative_matrix_m = 16;
    uint32_t cooperative_matrix_n = 16;
    uint32_t cooperative_matrix_k = 16;
    uint32_t optimal_workgroup_size = 256;
};

// Benchmark result structure
struct BenchmarkResult {
    uint32_t warmup_tokens = 0;
    uint32_t benchmark_tokens = 0;
    uint32_t iterations = 0;
    
    double avg_tokens_per_sec = 0.0;
    double min_tokens_per_sec = 0.0;
    double max_tokens_per_sec = 0.0;
    double std_dev_tokens_per_sec = 0.0;
    double avg_latency_ms = 0.0;
    double peak_vram_gb = 0.0;
    double cache_hit_rate = 0.0;
    double cache_size_mb = 0.0;
    
    std::unordered_map<std::string, double> power_mode_results;
    double memory_pressure_result = 0.0;
};

// Detailed benchmark statistics
struct BenchmarkStats {
    double avg_tokens_per_sec = 0.0;
    double min_tokens_per_sec = 0.0;
    double max_tokens_per_sec = 0.0;
    double std_dev_tokens_per_sec = 0.0;
    double avg_latency_ms = 0.0;
    double peak_vram_gb = 0.0;
    double avg_vram_usage_gb = 0.0;
    double fragmentation_ratio = 0.0;
    double cache_hit_rate = 0.0;
    uint64_t total_evictions = 0;
    double eviction_rate = 0.0;
    
    void print() const;
};

class VulkanSymbioteEngine {
public:
    VulkanSymbioteEngine(const Path& model_path);
    ~VulkanSymbioteEngine();

    std::string generate(const std::string& prompt, uint32_t max_tokens = 256, float temperature = 0.7f);
    std::vector<uint32_t> encode(const std::string& text);
    std::string decode(const std::vector<uint32_t>& tokens);

    const ModelConfig& config() const noexcept { return config_; }
    ModelConfig& config() noexcept { return config_; }

    VkDevice device() const noexcept { return device_; }
    VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    VkQueue compute_queue() const noexcept { return compute_queue_; }
    
    // Benchmark mode
    BenchmarkResult run_benchmark(uint32_t warmup_tokens = 10, uint32_t benchmark_tokens = 100, 
                                   uint32_t iterations = 3);
    
    // Detailed benchmark with full stats
    struct BenchmarkStats run_benchmark_detailed(uint32_t warmup_tokens, 
                                                  uint32_t benchmark_tokens,
                                                  uint32_t iterations);
    
    // Batched text generation
    std::vector<std::string> generate_text_batch(const std::vector<std::string>& prompts, 
                                                  uint32_t max_tokens_per_prompt = 256,
                                                  float temperature = 0.7f);
    
    // Memory statistics
    MemoryPoolStats get_vram_stats();
    double get_peak_vram_usage();
    
    // KV Cache management
    void clear_kv_cache();
    size_t get_kv_cache_memory_usage() const;

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = UINT32_MAX;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = nullptr;

    ModelConfig config_;
    Path model_path_;

    std::unique_ptr<PackManager> pack_manager_;
    std::unique_ptr<VitalityOracle> vitality_oracle_;
    std::unique_ptr<ShaderRuntime> shader_runtime_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<GGUFLoader> gguf_loader_;
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    std::unique_ptr<PowerManager> power_manager_;

    std::vector<float> hidden_states_;
    std::vector<uint32_t> token_sequence_;
    uint32_t current_position_ = 0;
    
    // Performance monitoring
    PerformanceMetrics performance_metrics_;
    DeviceCapabilities device_caps_;

    ExpectedVoid initialize_vulkan();
    ExpectedVoid load_model();
    ExpectedVoid create_pipelines();
    ExpectedVoid destroy_resources();

    Expected<std::vector<float>> embed_tokens(const std::vector<uint32_t>& tokens);
    Expected<std::vector<float>> forward_layer(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> attention(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> attention_with_cache(const std::vector<float>& hidden, uint32_t layer_idx,
                                                       const class KVCacheManager& kv_cache_mgr, uint32_t cache_idx);
    Expected<std::vector<float>> feed_forward(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> rms_norm(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> apply_rope(const std::vector<float>& hidden, uint32_t position);
    Expected<std::vector<float>> final_projection(const std::vector<float>& hidden);

    void schedule_prefetch(uint32_t current_layer, uint32_t lookahead);
    void evict_low_priority();

    ExpectedVoid upload_to_gpu(const std::vector<float>& data, VkBuffer buffer);
    ExpectedVoid download_from_gpu(VkBuffer buffer, std::vector<float>& data);
    
    // GPU helper methods
    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer cmd_buffer);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    
    // Power management
    enum class PowerProfile { HIGH_PERFORMANCE, BALANCED, POWER_SAVER };
    
    void detect_power_source();
    bool is_on_battery() const noexcept { return on_battery_; }
    void set_power_profile(PowerProfile profile);
    PowerProfile get_power_profile() const noexcept { return power_profile_; }
    void apply_power_settings();
    
    // Throttling controls
    void set_workgroup_size(uint32_t x, uint32_t y, uint32_t z);
    void get_workgroup_size(uint32_t& x, uint32_t& y, uint32_t& z) const noexcept;
    void set_prefetch_lookahead(uint32_t layers) { prefetch_lookahead_ = layers; }
    uint32_t get_prefetch_lookahead() const noexcept { return prefetch_lookahead_; }
    void enable_profiling(bool enable) { profiling_enabled_ = enable; }
    bool is_profiling_enabled() const noexcept { return profiling_enabled_; }
    
public:
    // Performance monitoring
    const PerformanceMetrics& get_performance_metrics() const { return performance_metrics_; }
    void reset_performance_metrics() { performance_metrics_.reset(); }
    
private:
    // Power management state
    bool on_battery_ = false;
    PowerProfile power_profile_ = PowerProfile::BALANCED;
    uint32_t workgroup_size_x_ = 256;
    uint32_t workgroup_size_y_ = 1;
    uint32_t workgroup_size_z_ = 1;
    uint32_t prefetch_lookahead_ = 3;
    bool profiling_enabled_ = true;
    uint32_t battery_check_interval_ms_ = 5000;
    uint64_t last_battery_check_ = 0;
    
    void check_battery_status();
    float read_battery_capacity();
    bool read_ac_connected();

    // Benchmark helpers
    std::unordered_map<std::string, double> test_power_modes(uint32_t tokens);
    double test_memory_pressure(uint32_t tokens);
    void save_benchmark_results_json(const BenchmarkResult& result, const std::string& filename);
    double calculate_mean(const std::vector<double>& samples);
    double calculate_std_dev(const std::vector<double>& samples);
    uint32_t sample_token(const std::vector<float>& logits, float temperature);
    uint32_t greedy_sampling(const float* logits, uint32_t size);
    void top_k_sampling(const float* logits, uint32_t vocab_size, uint32_t k, float temperature,
                       std::vector<uint32_t>& tokens, std::vector<float>& probs, float* workspace);
    
    // KV cache helpers
    void append_kv_cache(uint32_t layer_idx, const std::vector<float>& key, const std::vector<float>& value);
    Expected<std::vector<float>> forward_layer_with_kv(const std::vector<float>& hidden, uint32_t layer_idx);
    
    // ============================================================================
    // Weight Binding System - Connects loaded model weights to GPU compute
    // ============================================================================
    
public:
    // Weight buffer handle for GPU resident weights
    struct WeightBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        uint64_t size = 0;
        std::string tensor_name;
        bool is_loaded = false;
        uint64_t last_used = 0;
    };
    
    // Layer weight set - all weights for a transformer layer
    struct LayerWeights {
        // Attention weights
        WeightBuffer attn_q;
        WeightBuffer attn_k;
        WeightBuffer attn_v;
        WeightBuffer attn_o;
        WeightBuffer attn_q_bias;
        WeightBuffer attn_k_bias;
        WeightBuffer attn_v_bias;
        WeightBuffer attn_o_bias;
        
        // Feed-forward weights
        WeightBuffer ffn_gate;
        WeightBuffer ffn_up;
        WeightBuffer ffn_down;
        WeightBuffer ffn_gate_bias;
        WeightBuffer ffn_up_bias;
        WeightBuffer ffn_down_bias;
        
        // Normalization weights
        WeightBuffer norm_attn_gamma;
        WeightBuffer norm_attn_beta;
        WeightBuffer norm_ffn_gamma;
        WeightBuffer norm_ffn_beta;
        
        bool is_loaded = false;
        uint32_t layer_idx = 0;
    };
    
    // Embedding and output weights
    struct EmbeddingWeights {
        WeightBuffer token_embedding;
        WeightBuffer position_embedding;
        WeightBuffer output_projection;
        WeightBuffer output_bias;
        bool is_loaded = false;
    };
    
private:
    
    // Weight loading and binding
    ExpectedVoid load_layer_weights(uint32_t layer_idx);
    ExpectedVoid load_all_weights();
    void unload_layer_weights(uint32_t layer_idx);
    
    // Load specific tensor weight from GGUF
    Expected<WeightBuffer> load_weight_buffer(const std::string& tensor_name);
    Expected<WeightBuffer> load_weight_buffer_cached(const std::string& tensor_name);
    
    // Bind weights to descriptor set for compute
    ExpectedVoid bind_weight_to_descriptor(VkDescriptorSet descriptor_set, 
                                           uint32_t binding,
                                           const WeightBuffer& weight);
    
    // Weight management
    void evict_unused_weights(uint64_t max_age_ns);
    size_t get_weights_memory_usage() const;
    void clear_all_weights();
    
    // Get layer weights (loads if needed)
    Expected<LayerWeights*> get_layer_weights(uint32_t layer_idx);
    Expected<EmbeddingWeights*> get_embedding_weights();
    
    // Tensor name helpers
    std::string get_tensor_name(const std::string& prefix, uint32_t layer_idx, 
                                const std::string& suffix);
    std::vector<std::string> get_layer_tensor_names(uint32_t layer_idx);
    
    // Async weight loading
    void prefetch_weights_async(uint32_t start_layer, uint32_t end_layer);
    
    // ============================================================================
    // Enhanced inference with weight binding
    // ============================================================================
    
    Expected<std::vector<float>> attention_with_weights(
        const std::vector<float>& hidden, 
        uint32_t layer_idx,
        const LayerWeights& weights);
    
    Expected<std::vector<float>> feed_forward_with_weights(
        const std::vector<float>& hidden,
        uint32_t layer_idx, 
        const LayerWeights& weights);
    
    Expected<std::vector<float>> rms_norm_with_weights(
        const std::vector<float>& hidden,
        const WeightBuffer& gamma,
        const WeightBuffer* beta);
    
    Expected<std::vector<float>> embed_tokens_with_weights(
        const std::vector<uint32_t>& tokens);
    
    Expected<std::vector<float>> final_projection_with_weights(
        const std::vector<float>& hidden);
    
    // Time helpers
    static uint64_t get_current_time_ns();
};

} // namespace vk_symbiote
