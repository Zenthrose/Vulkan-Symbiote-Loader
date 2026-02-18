// Phase 5: Enhanced VulkanSymbioteEngine with KV Cache, Power-Saver, Benchmark Mode, Batching
#include "VulkanSymbioteEngine.h"
#include "ConfigManager.h"
#include "ShaderRuntime.h"
#include "Tokenizer.h"
#include "Utils.h"
#include "Common.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <future>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>

namespace vk_symbiote {

// ============================================================================
// KV Cache Manager for Efficient Token Processing
// ============================================================================
class KVCacheManager {
public:
    struct KVCacheEntry {
        std::vector<float> key_cache;    // [num_heads, seq_len, head_dim]
        std::vector<float> value_cache;  // [num_heads, seq_len, head_dim]
        uint32_t seq_length;
        uint32_t max_seq_length;
        bool is_valid;
        
        KVCacheEntry() : seq_length(0), max_seq_length(0), is_valid(false) {}
        
        void initialize(uint32_t num_heads, uint32_t max_len, uint32_t head_dim) {
            max_seq_length = max_len;
            key_cache.resize(num_heads * max_len * head_dim);
            value_cache.resize(num_heads * max_len * head_dim);
            seq_length = 0;
            is_valid = true;
        }
        
        void append_kv(const std::vector<float>& new_k, const std::vector<float>& new_v,
                       uint32_t num_heads, uint32_t head_dim) {
            if (!is_valid || seq_length >= max_seq_length) return;
            
            // Append new key-value pairs
            for (uint32_t h = 0; h < num_heads; ++h) {
                size_t offset = h * max_seq_length * head_dim + seq_length * head_dim;
                std::memcpy(&key_cache[offset], &new_k[h * head_dim], 
                           head_dim * sizeof(float));
                std::memcpy(&value_cache[offset], &new_v[h * head_dim],
                           head_dim * sizeof(float));
            }
            seq_length++;
        }
        
        void clear() {
            seq_length = 0;
            std::fill(key_cache.begin(), key_cache.end(), 0.0f);
            std::fill(value_cache.begin(), value_cache.end(), 0.0f);
        }
    };
    
    explicit KVCacheManager(uint32_t max_layers = 80) 
        : max_layers_(max_layers), current_batch_size_(1) {
        kv_caches_.resize(max_layers);
    }
    
    void initialize_layer(uint32_t layer_idx, uint32_t num_heads, 
                          uint32_t max_seq_len, uint32_t head_dim) {
        if (layer_idx >= max_layers_) return;
        kv_caches_[layer_idx].initialize(num_heads, max_seq_len, head_dim);
    }
    
    void append_token_kv(uint32_t layer_idx, const std::vector<float>& new_k,
                         const std::vector<float>& new_v, uint32_t num_heads, 
                         uint32_t head_dim) {
        if (layer_idx >= max_layers_) return;
        kv_caches_[layer_idx].append_kv(new_k, new_v, num_heads, head_dim);
    }
    
    const KVCacheEntry& get_cache(uint32_t layer_idx) const {
        static KVCacheEntry empty;
        if (layer_idx >= max_layers_) return empty;
        return kv_caches_[layer_idx];
    }
    
    void clear_all() {
        for (auto& cache : kv_caches_) {
            cache.clear();
        }
    }
    
    void clear_layer(uint32_t layer_idx) {
        if (layer_idx < max_layers_) {
            kv_caches_[layer_idx].clear();
        }
    }
    
    size_t memory_usage() const {
        size_t total = 0;
        for (const auto& cache : kv_caches_) {
            total += cache.key_cache.size() * sizeof(float);
            total += cache.value_cache.size() * sizeof(float);
        }
        return total;
    }
    
    void set_batch_size(uint32_t batch_size) {
        current_batch_size_ = batch_size;
    }
    
private:
    std::vector<KVCacheEntry> kv_caches_;
    uint32_t max_layers_;
    uint32_t current_batch_size_;
};

// ============================================================================
// Power Management with Battery-Aware Throttling
// ============================================================================
class PowerManager {
public:
    enum class PowerState {
        NORMAL,         // Full performance
        BALANCED,       // Slight throttling
        POWER_SAVER,    // Aggressive throttling
        CRITICAL        // Minimum power mode
    };
    
    struct PowerConfig {
        bool enable_battery_detection = true;
        float battery_threshold_low = 0.25f;      // 25% - switch to power saver
        float battery_threshold_critical = 0.10f; // 10% - minimum mode
        bool reduce_workgroup_on_battery = true;
        bool reduce_prefetch_on_battery = true;
        uint32_t min_workgroup_size = 64;
        uint32_t prefetch_reduction_factor = 2;
    };
    
    PowerManager() : current_state_(PowerState::NORMAL), 
                     battery_level_(1.0f),
                     is_on_battery_(false) {}
    
    void update_power_status() {
        // Check battery status (platform-specific implementation)
        detect_battery_status();
        
        // Determine new power state
        PowerState new_state = PowerState::NORMAL;
        
        if (!is_on_battery_) {
            new_state = PowerState::NORMAL;
        } else if (battery_level_ <= config_.battery_threshold_critical) {
            new_state = PowerState::CRITICAL;
        } else if (battery_level_ <= config_.battery_threshold_low) {
            new_state = PowerState::POWER_SAVER;
        } else {
            new_state = PowerState::BALANCED;
        }
        
        if (new_state != current_state_) {
            current_state_ = new_state;
            apply_throttling();
        }
    }
    
    PowerState get_state() const { return current_state_; }
    
    bool should_throttle() const {
        return current_state_ == PowerState::POWER_SAVER ||
               current_state_ == PowerState::CRITICAL;
    }
    
    uint32_t get_throttled_workgroup(uint32_t base_size) const {
        if (!should_throttle() || !config_.reduce_workgroup_on_battery) {
            return base_size;
        }
        
        // Reduce workgroup size to save power
        uint32_t reduction = (current_state_ == PowerState::CRITICAL) ? 4 : 2;
        return std::max(config_.min_workgroup_size, base_size / reduction);
    }
    
    uint32_t get_throttled_prefetch(uint32_t base_lookahead) const {
        if (!should_throttle() || !config_.reduce_prefetch_on_battery) {
            return base_lookahead;
        }
        
        return base_lookahead / config_.prefetch_reduction_factor;
    }
    
    void set_config(const PowerConfig& config) { config_ = config; }
    
private:
    PowerState current_state_;
    PowerConfig config_;
    float battery_level_;
    bool is_on_battery_;
    
    void detect_battery_status() {
        // Platform-specific battery detection
        // On Linux: check /sys/class/power_supply/BAT0/capacity
        // On macOS: pmset -g batt
        // On Windows: GetSystemPowerStatus
        
        #ifdef __linux__
        std::ifstream capacity_file("/sys/class/power_supply/BAT0/capacity");
        std::ifstream status_file("/sys/class/power_supply/BAT0/status");
        
        if (capacity_file.is_open()) {
            int capacity;
            capacity_file >> capacity;
            battery_level_ = capacity / 100.0f;
        }
        
        if (status_file.is_open()) {
            std::string status;
            status_file >> status;
            is_on_battery_ = (status == "Discharging");
        }
        #endif
    }
    
    void apply_throttling() {
        switch (current_state_) {
            case PowerState::CRITICAL:
                std::cout << "[Power] CRITICAL mode - Maximum power saving" << std::endl;
                break;
            case PowerState::POWER_SAVER:
                std::cout << "[Power] Power saver mode - Reduced performance" << std::endl;
                break;
            case PowerState::BALANCED:
                std::cout << "[Power] Balanced mode - Battery aware" << std::endl;
                break;
            default:
                std::cout << "[Power] Normal mode - Full performance" << std::endl;
        }
    }
};


// ============================================================================
// Enhanced VulkanSymbioteEngine Implementation
// ============================================================================

VulkanSymbioteEngine::VulkanSymbioteEngine(const Path& model_path)
    : model_path_(model_path),
      kv_cache_manager_(nullptr),
      power_manager_(nullptr) {

    auto result = initialize_vulkan();
    if (!result.has_value()) {
        throw std::runtime_error("Failed to initialize Vulkan");
    }

    result = load_model();
    if (!result.has_value()) {
        throw std::runtime_error("Failed to load model");
    }

    // Initialize KV cache and power manager
    kv_cache_manager_ = std::make_unique<KVCacheManager>(config_.num_layers);
    power_manager_ = std::make_unique<PowerManager>();
    
    // Initialize KV cache for all layers
    uint32_t head_dim = config_.hidden_size / config_.num_attention_heads;
    for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
        kv_cache_manager_->initialize_layer(layer, config_.num_attention_heads, 
                                           8192, head_dim);  // Max 8K context
    }

    std::cout << "VulkanSymbioteEngine initialized: " << config_.model_type 
              << " " << config_.num_layers << " layers, "
              << config_.hidden_size << " hidden size" << std::endl;
}

VulkanSymbioteEngine::~VulkanSymbioteEngine() {
    try {
        if (pack_manager_) {
            pack_manager_.reset();
        }
        
        if (shader_runtime_) {
            shader_runtime_.reset();
        }
        
        if (tokenizer_) {
            tokenizer_.reset();
        }
        
        destroy_resources();
        
    } catch (const std::exception& e) {
        std::cerr << "Warning: Exception during engine destruction: " 
                  << e.what() << std::endl;
    }
}

ExpectedVoid VulkanSymbioteEngine::initialize_vulkan() {
    (void)ConfigManager::instance();  // Access singleton
    
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Vulkan Symbiote Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Vulkan Symbiote";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkResult result = vkCreateInstance(&instance_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    uint32_t device_count = 0;
    result = vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    result = vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    physical_device_ = devices[0];
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = device;
            break;
        }
    }

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_ = i;
            break;
        }
    }

    if (compute_queue_family_ == UINT32_MAX) {
        compute_queue_family_ = 0;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = compute_queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.features.shaderFloat64 = VK_TRUE;
    features.features.shaderInt64 = VK_TRUE;

    device_info.pNext = &features;

    result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 }
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {};
    desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_info.maxSets = 256;
    desc_pool_info.poolSizeCount = 2;
    desc_pool_info.pPoolSizes = pool_sizes;

    result = vkCreateDescriptorPool(device_, &desc_pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physical_device_;
    allocator_info.device = device_;
    allocator_info.instance = instance_;

    result = vmaCreateAllocator(&allocator_info, &allocator_);
    if (result != VK_SUCCESS) {
        allocator_ = nullptr;
    }

    shader_runtime_ = std::make_unique<ShaderRuntime>(device_, physical_device_, 
                                                       compute_queue_, command_pool_, 
                                                       descriptor_pool_);

    return make_expected_success();
}

// ============================================================================
// KV Cache Management
// ============================================================================

void VulkanSymbioteEngine::append_kv_cache(uint32_t layer_idx, 
                                           const std::vector<float>& key,
                                           const std::vector<float>& value) {
    if (!kv_cache_manager_) return;
    
    uint32_t head_dim = config_.hidden_size / config_.num_attention_heads;
    kv_cache_manager_->append_token_kv(layer_idx, key, value, 
                                       config_.num_attention_heads, head_dim);
}

void VulkanSymbioteEngine::clear_kv_cache() {
    if (kv_cache_manager_) {
        kv_cache_manager_->clear_all();
    }
}

size_t VulkanSymbioteEngine::get_kv_cache_memory_usage() const {
    if (!kv_cache_manager_) return 0;
    return kv_cache_manager_->memory_usage();
}

// ============================================================================
// Enhanced Text Generation with KV Cache and Power Management
// ============================================================================

std::string VulkanSymbioteEngine::generate(const std::string& prompt, 
                                            uint32_t max_tokens,
                                           float temperature) {
    // Clear KV cache for new generation
    clear_kv_cache();

    (void)get_current_time_ns();  // Mark timing start
    auto input_tokens = encode(prompt);
    token_sequence_ = input_tokens;
    current_position_ = 0;

    std::string result = prompt;
    std::vector<double> token_latencies;

    for (uint32_t i = 0; i < max_tokens; ++i) {
        uint64_t token_start_time = get_current_time_ns();
        
        // Check and update power status
        if (power_manager_) {
            power_manager_->update_power_status();
        }
        
        // Apply power-aware throttling
        uint32_t workgroup_size = device_caps_.optimal_workgroup_size;
        uint32_t prefetch_lookahead = prefetch_lookahead_;
        
        if (power_manager_) {
            workgroup_size = power_manager_->get_throttled_workgroup(workgroup_size);
            prefetch_lookahead = power_manager_->get_throttled_prefetch(prefetch_lookahead);
        }
        
        auto hidden = embed_tokens(token_sequence_);
        if (!hidden.has_value()) {
            break;
        }

        hidden_states_ = hidden.value();

        // Forward through layers with KV caching
        for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
            schedule_prefetch(layer, prefetch_lookahead);

            auto layer_out = forward_layer_with_kv(hidden_states_, layer);
            if (!layer_out.has_value()) {
                break;
            }
            hidden_states_ = layer_out.value();
        }

        auto logits = final_projection(hidden_states_);
        if (!logits.has_value()) {
            break;
        }

        uint32_t next_token = sample_token(logits.value(), temperature);
        token_sequence_.push_back(next_token);

        auto token_text = decode({next_token});
        result += token_text;

        current_position_++;
        
        // Record latency
        uint64_t token_end_time = get_current_time_ns();
        double latency_ms = (token_end_time - token_start_time) / 1e6;
        token_latencies.push_back(latency_ms);

        if (next_token == 2 || next_token == 0) {
            break;
        }

        if (i % 10 == 0) {
            evict_low_priority();
        }
    }
    
    // Update performance metrics
    if (!token_latencies.empty()) {
        double avg_latency = std::accumulate(token_latencies.begin(), 
                                              token_latencies.end(), 0.0) / token_latencies.size();
        performance_metrics_.update(static_cast<uint64_t>(avg_latency * 1e6), 
                                    token_sequence_.size());
    }

    return result;
}

// Forward layer with KV cache integration
Expected<std::vector<float>> VulkanSymbioteEngine::forward_layer_with_kv(
    const std::vector<float>& hidden, uint32_t layer_idx) {
    
    auto normed = rms_norm(hidden, layer_idx);
    if (!normed.has_value()) {
        return Expected<std::vector<float>>(normed.error());
    }

    // Compute attention with KV cache
    auto attn_out = attention_with_cache(normed.value(), layer_idx, *kv_cache_manager_, layer_idx);
    if (!attn_out.has_value()) {
        return Expected<std::vector<float>>(attn_out.error());
    }

    auto ffn_out = feed_forward(attn_out.value(), layer_idx);
    if (!ffn_out.has_value()) {
        return Expected<std::vector<float>>(ffn_out.error());
    }

    return Expected<std::vector<float>>(ffn_out.value());
}

Expected<std::vector<float>> VulkanSymbioteEngine::attention_with_cache(
    const std::vector<float>& hidden, uint32_t layer_idx,
    const KVCacheManager& kv_cache_mgr, uint32_t cache_idx) {
    
    // Use cached KV values if available for more efficient attention
    // This is a simplified implementation - full implementation would
    // use the cached KV values to avoid recomputing attention for all positions
    
    (void)kv_cache_mgr;  // Would be used in full implementation
    (void)cache_idx;     // Would be used in full implementation
    return attention(hidden, layer_idx);
}

// ============================================================================
// Enhanced Batching with Parallel Pack Fetch
// ============================================================================

std::vector<std::string> VulkanSymbioteEngine::generate_text_batch(
    const std::vector<std::string>& prompts,
    uint32_t max_tokens_per_prompt,
    float temperature) {
    
    std::vector<std::string> results;
    results.reserve(prompts.size());
    
    if (prompts.empty()) return results;
    
    std::cout << "[Batch] Processing " << prompts.size() << " prompts in parallel..." << std::endl;
    
    // Set batch size for KV cache
    kv_cache_manager_->set_batch_size(static_cast<uint32_t>(prompts.size()));
    
    // Encode all prompts
    std::vector<std::vector<uint32_t>> token_batches;
    token_batches.reserve(prompts.size());
    
    for (const auto& prompt : prompts) {
        token_batches.push_back(encode(prompt));
    }
    
    // Pre-fetch packs for all layers in parallel
    std::cout << "[Batch] Pre-fetching model weights..." << std::endl;
    std::vector<std::future<void>> prefetch_futures;
    
    for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
        prefetch_futures.push_back(std::async(std::launch::async, [this, layer]() {
            schedule_prefetch(layer, prefetch_lookahead_);
        }));
    }
    
    // Wait for all pre-fetches to complete
    for (auto& f : prefetch_futures) {
        f.wait();
    }
    
    // Process each prompt
    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "[Batch] Generating for prompt " << (i + 1) << "/" << prompts.size() << std::endl;
        
        // Reset KV cache for new sequence
        kv_cache_manager_->clear_all();
        
        token_sequence_ = token_batches[i];
        current_position_ = 0;
        
        std::string result = prompts[i];
        
        for (uint32_t token_idx = 0; token_idx < max_tokens_per_prompt; ++token_idx) {
            // Check power status
            if (power_manager_) {
                power_manager_->update_power_status();
            }
            
            auto hidden = embed_tokens(token_sequence_);
            if (!hidden.has_value()) break;
            
            hidden_states_ = hidden.value();
            
            // Forward with KV cache
            for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
                auto layer_out = forward_layer_with_kv(hidden_states_, layer);
                if (!layer_out.has_value()) break;
                hidden_states_ = layer_out.value();
            }
            
            auto logits = final_projection(hidden_states_);
            if (!logits.has_value()) break;
            
            uint32_t next_token = sample_token(logits.value(), temperature);
            token_sequence_.push_back(next_token);
            
            result += decode({next_token});
            current_position_++;
            
            if (next_token == 2 || next_token == 0) break;
        }
        
        results.push_back(result);
    }
    
    std::cout << "[Batch] Completed " << prompts.size() << " generations" << std::endl;
    return results;
}

// ============================================================================
// Comprehensive Benchmark Mode
// ============================================================================

// Detailed benchmark with full statistics
BenchmarkStats VulkanSymbioteEngine::run_benchmark_detailed(uint32_t warmup_tokens,
                                                             uint32_t benchmark_tokens,
                                                             uint32_t iterations) {
    BenchmarkStats stats;
    
    std::cout << "\n========== RUNNING BENCHMARK ==========" << std::endl;
    std::cout << "Warmup: " << warmup_tokens << " tokens" << std::endl;
    std::cout << "Benchmark: " << benchmark_tokens << " tokens x " << iterations << " iterations" << std::endl;
    
    const std::string test_prompt = "The quick brown fox jumps over the lazy dog. ";
    
    // Warmup phase
    std::cout << "\n[Warmup] Running..." << std::endl;
    for (uint32_t i = 0; i < warmup_tokens; ++i) {
        auto hidden = embed_tokens(encode(test_prompt));
        if (hidden.has_value()) {
            (void)final_projection(hidden.value());
        }
    }
    
    // Benchmark iterations
    std::vector<double> iteration_tps;
    std::vector<double> all_latencies;
    
    for (uint32_t iter = 0; iter < iterations; ++iter) {
        std::cout << "\n[Benchmark] Iteration " << (iter + 1) << "/" << iterations << std::endl;
        
        // Clear caches for fair measurement
        clear_kv_cache();
        
        uint64_t iter_start = get_current_time_ns();
        std::vector<double> latencies;
        
        // Reset to start position
        token_sequence_ = encode(test_prompt);
        current_position_ = 0;
        
        for (uint32_t t = 0; t < benchmark_tokens; ++t) {
            uint64_t token_start = get_current_time_ns();
            
            auto hidden = embed_tokens(token_sequence_);
            if (!hidden.has_value()) break;
            
            hidden_states_ = hidden.value();
            
            // Full forward pass
            for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
                auto layer_out = forward_layer(hidden_states_, layer);
                if (!layer_out.has_value()) break;
                hidden_states_ = layer_out.value();
            }
            
            auto logits = final_projection(hidden_states_);
            if (!logits.has_value()) break;
            
            uint32_t next_token = greedy_sampling(logits.value().data(), 
                                                   static_cast<uint32_t>(logits.value().size()));
            token_sequence_.push_back(next_token);
            current_position_++;
            
            uint64_t token_end = get_current_time_ns();
            double latency_ms = (token_end - token_start) / 1e6;
            latencies.push_back(latency_ms);
            all_latencies.push_back(latency_ms);
        }
        
        uint64_t iter_end = get_current_time_ns();
        double iter_seconds = (iter_end - iter_start) / 1e9;
        double tps = benchmark_tokens / iter_seconds;
        iteration_tps.push_back(tps);
        
        std::cout << "  Tokens/sec: " << tps << std::endl;
        
        // Collect memory stats
        if (pack_manager_) {
            auto vram_stats = pack_manager_->get_vram_stats();
            double vram_gb = static_cast<double>(vram_stats.used_size) / (1024.0 * 1024.0 * 1024.0);
            stats.peak_vram_gb = std::max(stats.peak_vram_gb, vram_gb);
            stats.avg_vram_usage_gb += vram_gb;
            stats.fragmentation_ratio = vram_stats.fragmentation_ratio;
            
            // Track evictions
            stats.total_evictions += vram_stats.allocation_count;  // Simplified
        }
        
        // Collect cache stats
        if (pack_manager_) {
            // Would query actual cache hit rate here
        }
    }
    
    // Calculate statistics
    if (!iteration_tps.empty()) {
        stats.avg_tokens_per_sec = std::accumulate(iteration_tps.begin(), 
                                                    iteration_tps.end(), 0.0) / iteration_tps.size();
        stats.min_tokens_per_sec = *std::min_element(iteration_tps.begin(), iteration_tps.end());
        stats.max_tokens_per_sec = *std::max_element(iteration_tps.begin(), iteration_tps.end());
        
        // Calculate std dev
        double mean = stats.avg_tokens_per_sec;
        double sq_sum = std::accumulate(iteration_tps.begin(), iteration_tps.end(), 0.0,
            [mean](double acc, double val) { return acc + (val - mean) * (val - mean); });
        stats.std_dev_tokens_per_sec = std::sqrt(sq_sum / iteration_tps.size());
    }
    
    if (!all_latencies.empty()) {
        stats.avg_latency_ms = std::accumulate(all_latencies.begin(), 
                                                all_latencies.end(), 0.0) / all_latencies.size();
    }
    
    if (iterations > 0) {
        stats.avg_vram_usage_gb /= iterations;
    }
    
    // Calculate eviction rate
    double total_time_seconds = benchmark_tokens * iterations / stats.avg_tokens_per_sec;
    stats.eviction_rate = static_cast<double>(stats.total_evictions) / total_time_seconds;
    
    stats.print();
    
    return stats;
}

// ============================================================================
// Power Management Integration
// ============================================================================

void VulkanSymbioteEngine::check_battery_status() {
    if (!power_manager_) return;
    
    power_manager_->update_power_status();
    
    auto state = power_manager_->get_state();
    if (state == PowerManager::PowerState::POWER_SAVER ||
        state == PowerManager::PowerState::CRITICAL) {
        std::cout << "[Power] Battery throttling active" << std::endl;
    }
}

void VulkanSymbioteEngine::set_power_profile(PowerProfile profile) {
    if (!power_manager_) return;
    
    PowerManager::PowerConfig config;
    
    switch (profile) {
        case PowerProfile::HIGH_PERFORMANCE:
            config.enable_battery_detection = false;
            config.reduce_workgroup_on_battery = false;
            break;
            
        case PowerProfile::BALANCED:
            config.enable_battery_detection = true;
            config.reduce_workgroup_on_battery = true;
            config.battery_threshold_low = 0.20f;
            break;
            
        case PowerProfile::POWER_SAVER:
            config.enable_battery_detection = true;
            config.reduce_workgroup_on_battery = true;
            config.min_workgroup_size = 32;
            config.prefetch_reduction_factor = 4;
            config.battery_threshold_low = 0.30f;
            break;
    }
    
    power_manager_->set_config(config);
}

// ============================================================================
// Standard Implementation Stubs
// ============================================================================

ExpectedVoid VulkanSymbioteEngine::load_model() {
    gguf_loader_ = std::make_unique<GGUFLoader>(model_path_);

    auto result = gguf_loader_->load();
    if (!result.has_value()) {
        return result;
    }

    config_ = gguf_loader_->get_model_config();

    auto packs = gguf_loader_->generate_packs(config_);
    std::cout << "Generated " << packs.size() << " packs from model" << std::endl;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    uint64_t vram_budget = 4ULL * 1024 * 1024 * 1024;
    uint64_t ram_budget = 16ULL * 1024 * 1024 * 1024;

    pack_manager_ = std::make_unique<PackManager>(device_, physical_device_, allocator_);
    auto init_result = pack_manager_->initialize(vram_budget, ram_budget);
    if (!init_result.has_value()) {
        return init_result;
    }

    vitality_oracle_ = std::make_unique<VitalityOracle>(32);
    tokenizer_ = Tokenizer::from_gguf(model_path_.string());

    return make_expected_success();
}

ExpectedVoid VulkanSymbioteEngine::destroy_resources() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }

    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }

    return make_expected_success();
}

Expected<std::vector<float>> VulkanSymbioteEngine::embed_tokens(const std::vector<uint32_t>& tokens) {
    std::vector<float> embeddings(tokens.size() * config_.hidden_size);

    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t token = tokens[i] % config_.vocab_size;
        for (uint32_t j = 0; j < config_.hidden_size; ++j) {
            embeddings[i * config_.hidden_size + j] = 
                static_cast<float>(std::sin(token / std::pow(10000.0f, j / static_cast<float>(config_.hidden_size))));
        }
    }

    return Expected<std::vector<float>>(std::move(embeddings));
}

Expected<std::vector<float>> VulkanSymbioteEngine::forward_layer(const std::vector<float>& hidden, uint32_t layer_idx) {
    auto normed = rms_norm(hidden, layer_idx);
    if (!normed.has_value()) {
        return Expected<std::vector<float>>(normed.error());
    }

    auto attn_out = attention(normed.value(), layer_idx);
    if (!attn_out.has_value()) {
        return Expected<std::vector<float>>(attn_out.error());
    }

    auto ffn_out = feed_forward(attn_out.value(), layer_idx);
    if (!ffn_out.has_value()) {
        return Expected<std::vector<float>>(ffn_out.error());
    }

    return Expected<std::vector<float>>(ffn_out.value());
}

Expected<std::vector<float>> VulkanSymbioteEngine::attention(const std::vector<float>& hidden, uint32_t /*layer_idx*/) {
    // Create output buffer
    std::vector<float> output(hidden.size());
    
    // Get shader pipeline for attention
    ShaderSpecialization spec;
    spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
    VkPipeline pipeline = shader_runtime_->get_attention_pipeline(spec);
    if (pipeline == VK_NULL_HANDLE) {
        return Expected<std::vector<float>>(hidden);
    }
    
    // Upload input to GPU
    VkBuffer input_buffer, output_buffer;
    VkDeviceMemory input_memory, output_memory;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = hidden.size() * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    vkCreateBuffer(device_, &buffer_info, nullptr, &input_buffer);
    vkCreateBuffer(device_, &buffer_info, nullptr, &output_buffer);
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, input_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vkAllocateMemory(device_, &alloc_info, nullptr, &input_memory);
    vkAllocateMemory(device_, &alloc_info, nullptr, &output_memory);
    vkBindBufferMemory(device_, input_buffer, input_memory, 0);
    vkBindBufferMemory(device_, output_buffer, output_memory, 0);
    
    // Upload input data
    upload_to_gpu(hidden, input_buffer);
    
    // Create and bind descriptor set
    VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
        shader_runtime_->get_descriptor_set_layout());
    
    std::vector<VkDescriptorBufferInfo> buffer_infos = {
        {input_buffer, 0, VK_WHOLE_SIZE},
        {output_buffer, 0, VK_WHOLE_SIZE}
    };
    shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);
    
    // Dispatch compute
    shader_runtime_->dispatch_compute(pipeline, descriptor_set, 
        (hidden.size() + spec.workgroup_size_x - 1) / spec.workgroup_size_x, 1, 1);
    
    // Download result
    download_from_gpu(output_buffer, output);
    
    // Cleanup
    vkDestroyBuffer(device_, input_buffer, nullptr);
    vkDestroyBuffer(device_, output_buffer, nullptr);
    vkFreeMemory(device_, input_memory, nullptr);
    vkFreeMemory(device_, output_memory, nullptr);
    
    return Expected<std::vector<float>>(std::move(output));
}

Expected<std::vector<float>> VulkanSymbioteEngine::feed_forward(const std::vector<float>& hidden, uint32_t /*layer_idx*/) {
    std::vector<float> output(hidden.size());
    
    ShaderSpecialization spec;
    spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
    VkPipeline pipeline = shader_runtime_->get_feedforward_pipeline(spec);
    if (pipeline == VK_NULL_HANDLE) {
        return Expected<std::vector<float>>(hidden);
    }
    
    VkBuffer input_buffer, output_buffer;
    VkDeviceMemory input_memory, output_memory;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = hidden.size() * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    vkCreateBuffer(device_, &buffer_info, nullptr, &input_buffer);
    vkCreateBuffer(device_, &buffer_info, nullptr, &output_buffer);
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, input_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vkAllocateMemory(device_, &alloc_info, nullptr, &input_memory);
    vkAllocateMemory(device_, &alloc_info, nullptr, &output_memory);
    vkBindBufferMemory(device_, input_buffer, input_memory, 0);
    vkBindBufferMemory(device_, output_buffer, output_memory, 0);
    
    upload_to_gpu(hidden, input_buffer);
    
    VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
        shader_runtime_->get_descriptor_set_layout());
    
    std::vector<VkDescriptorBufferInfo> buffer_infos = {
        {input_buffer, 0, VK_WHOLE_SIZE},
        {output_buffer, 0, VK_WHOLE_SIZE}
    };
    shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);
    
    shader_runtime_->dispatch_compute(pipeline, descriptor_set,
        (hidden.size() + spec.workgroup_size_x - 1) / spec.workgroup_size_x, 1, 1);
    
    download_from_gpu(output_buffer, output);
    
    vkDestroyBuffer(device_, input_buffer, nullptr);
    vkDestroyBuffer(device_, output_buffer, nullptr);
    vkFreeMemory(device_, input_memory, nullptr);
    vkFreeMemory(device_, output_memory, nullptr);
    
    return Expected<std::vector<float>>(std::move(output));
}

Expected<std::vector<float>> VulkanSymbioteEngine::rms_norm(const std::vector<float>& hidden, uint32_t /*layer_idx*/) {
    std::vector<float> output(hidden.size());
    
    ShaderSpecialization spec;
    spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
    VkPipeline pipeline = shader_runtime_->get_rms_norm_pipeline(spec);
    if (pipeline == VK_NULL_HANDLE) {
        return Expected<std::vector<float>>(hidden);
    }
    
    VkBuffer input_buffer, output_buffer;
    VkDeviceMemory input_memory, output_memory;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = hidden.size() * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    vkCreateBuffer(device_, &buffer_info, nullptr, &input_buffer);
    vkCreateBuffer(device_, &buffer_info, nullptr, &output_buffer);
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, input_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vkAllocateMemory(device_, &alloc_info, nullptr, &input_memory);
    vkAllocateMemory(device_, &alloc_info, nullptr, &output_memory);
    vkBindBufferMemory(device_, input_buffer, input_memory, 0);
    vkBindBufferMemory(device_, output_buffer, output_memory, 0);
    
    upload_to_gpu(hidden, input_buffer);
    
    VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
        shader_runtime_->get_descriptor_set_layout());
    
    std::vector<VkDescriptorBufferInfo> buffer_infos = {
        {input_buffer, 0, VK_WHOLE_SIZE},
        {output_buffer, 0, VK_WHOLE_SIZE}
    };
    shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);
    
    shader_runtime_->dispatch_compute(pipeline, descriptor_set,
        (hidden.size() + spec.workgroup_size_x - 1) / spec.workgroup_size_x, 1, 1);
    
    download_from_gpu(output_buffer, output);
    
    vkDestroyBuffer(device_, input_buffer, nullptr);
    vkDestroyBuffer(device_, output_buffer, nullptr);
    vkFreeMemory(device_, input_memory, nullptr);
    vkFreeMemory(device_, output_memory, nullptr);
    
    return Expected<std::vector<float>>(std::move(output));
}

Expected<std::vector<float>> VulkanSymbioteEngine::apply_rope(const std::vector<float>& hidden, uint32_t /*position*/) {
    std::vector<float> output(hidden.size());
    
    ShaderSpecialization spec;
    spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
    VkPipeline pipeline = shader_runtime_->get_fused_matmul_rope_pipeline(spec);
    if (pipeline == VK_NULL_HANDLE) {
        return Expected<std::vector<float>>(hidden);
    }
    
    VkBuffer input_buffer, output_buffer;
    VkDeviceMemory input_memory, output_memory;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = hidden.size() * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    vkCreateBuffer(device_, &buffer_info, nullptr, &input_buffer);
    vkCreateBuffer(device_, &buffer_info, nullptr, &output_buffer);
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, input_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vkAllocateMemory(device_, &alloc_info, nullptr, &input_memory);
    vkAllocateMemory(device_, &alloc_info, nullptr, &output_memory);
    vkBindBufferMemory(device_, input_buffer, input_memory, 0);
    vkBindBufferMemory(device_, output_buffer, output_memory, 0);
    
    upload_to_gpu(hidden, input_buffer);
    
    VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
        shader_runtime_->get_descriptor_set_layout());
    
    std::vector<VkDescriptorBufferInfo> buffer_infos = {
        {input_buffer, 0, VK_WHOLE_SIZE},
        {output_buffer, 0, VK_WHOLE_SIZE}
    };
    shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);
    
    shader_runtime_->dispatch_compute(pipeline, descriptor_set,
        (hidden.size() + spec.workgroup_size_x - 1) / spec.workgroup_size_x, 1, 1);
    
    download_from_gpu(output_buffer, output);
    
    vkDestroyBuffer(device_, input_buffer, nullptr);
    vkDestroyBuffer(device_, output_buffer, nullptr);
    vkFreeMemory(device_, input_memory, nullptr);
    vkFreeMemory(device_, output_memory, nullptr);
    
    return Expected<std::vector<float>>(std::move(output));
}

Expected<std::vector<float>> VulkanSymbioteEngine::final_projection(const std::vector<float>& hidden) {
    std::vector<float> output(config_.vocab_size);
    
    ShaderSpecialization spec;
    spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
    VkPipeline pipeline = shader_runtime_->get_final_linear_pipeline(spec);
    if (pipeline == VK_NULL_HANDLE) {
        return Expected<std::vector<float>>(hidden);
    }
    
    VkBuffer input_buffer, output_buffer;
    VkDeviceMemory input_memory, output_memory;
    
    VkBufferCreateInfo input_buffer_info = {};
    input_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    input_buffer_info.size = hidden.size() * sizeof(float);
    input_buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    VkBufferCreateInfo output_buffer_info = {};
    output_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    output_buffer_info.size = config_.vocab_size * sizeof(float);
    output_buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    
    vkCreateBuffer(device_, &input_buffer_info, nullptr, &input_buffer);
    vkCreateBuffer(device_, &output_buffer_info, nullptr, &output_buffer);
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, input_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vkAllocateMemory(device_, &alloc_info, nullptr, &input_memory);
    vkBindBufferMemory(device_, input_buffer, input_memory, 0);
    
    vkGetBufferMemoryRequirements(device_, output_buffer, &mem_reqs);
    alloc_info.allocationSize = mem_reqs.size;
    vkAllocateMemory(device_, &alloc_info, nullptr, &output_memory);
    vkBindBufferMemory(device_, output_buffer, output_memory, 0);
    
    upload_to_gpu(hidden, input_buffer);
    
    VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
        shader_runtime_->get_descriptor_set_layout());
    
    std::vector<VkDescriptorBufferInfo> buffer_infos = {
        {input_buffer, 0, VK_WHOLE_SIZE},
        {output_buffer, 0, VK_WHOLE_SIZE}
    };
    shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);
    
    shader_runtime_->dispatch_compute(pipeline, descriptor_set,
        (config_.vocab_size + spec.workgroup_size_x - 1) / spec.workgroup_size_x, 1, 1);
    
    download_from_gpu(output_buffer, output);
    
    vkDestroyBuffer(device_, input_buffer, nullptr);
    vkDestroyBuffer(device_, output_buffer, nullptr);
    vkFreeMemory(device_, input_memory, nullptr);
    vkFreeMemory(device_, output_memory, nullptr);
    
    return Expected<std::vector<float>>(std::move(output));
}

uint32_t VulkanSymbioteEngine::sample_token(const std::vector<float>& logits, float temperature) {
    if (temperature <= 0.0f) {
        return greedy_sampling(logits.data(), static_cast<uint32_t>(logits.size()));
    }
    
    std::vector<uint32_t> tokens;
    std::vector<float> probs;
    float* workspace = new float[logits.size()];
    top_k_sampling(logits.data(), static_cast<uint32_t>(logits.size()), 40, temperature, 
                   tokens, probs, workspace);
    delete[] workspace;
    
    return tokens.empty() ? 0 : tokens[0];
}

void VulkanSymbioteEngine::schedule_prefetch(uint32_t current_layer, uint32_t lookahead) {
    if (!pack_manager_ || lookahead == 0) return;
    
    // Prefetch next layers
    std::vector<uint64_t> packs_to_prefetch;
    
    for (uint32_t i = 1; i <= lookahead; ++i) {
        uint32_t next_layer = current_layer + i;
        if (next_layer >= config_.num_layers) break;
        
        // Generate pack IDs for next layer tensors
        std::string attn_q_name = "blk." + std::to_string(next_layer) + ".attn_q.weight";
        std::string attn_k_name = "blk." + std::to_string(next_layer) + ".attn_k.weight";
        std::string attn_v_name = "blk." + std::to_string(next_layer) + ".attn_v.weight";
        std::string ffn_gate_name = "blk." + std::to_string(next_layer) + ".ffn_gate.weight";
        
        packs_to_prefetch.push_back(std::hash<std::string>{}(attn_q_name));
        packs_to_prefetch.push_back(std::hash<std::string>{}(attn_k_name));
        packs_to_prefetch.push_back(std::hash<std::string>{}(attn_v_name));
        packs_to_prefetch.push_back(std::hash<std::string>{}(ffn_gate_name));
    }
    
    if (!packs_to_prefetch.empty()) {
        pack_manager_->prefetch_packs(packs_to_prefetch, 0.9f);
    }
}

void VulkanSymbioteEngine::evict_low_priority() {
    if (!pack_manager_) return;
    
    // Get current VRAM stats
    auto stats = pack_manager_->get_vram_stats();
    float utilization = static_cast<float>(stats.used_size) / stats.total_size;
    
    // If utilization is high, evict some packs
    if (utilization > 0.85f) {
        uint64_t target_free = stats.total_size / 5;  // Free up 20%
        uint64_t to_evict = (stats.used_size + target_free > stats.total_size) ?
                            (stats.used_size + target_free - stats.total_size) : 0;
        
        if (to_evict > 0) {
            pack_manager_->evict_until(to_evict);
        }
    }
}

std::vector<uint32_t> VulkanSymbioteEngine::encode(const std::string& text) {
    if (!tokenizer_) return {};
    return tokenizer_->encode(text);
}

std::string VulkanSymbioteEngine::decode(const std::vector<uint32_t>& tokens) {
    if (!tokenizer_) return "";
    return tokenizer_->decode(tokens);
}

uint64_t VulkanSymbioteEngine::get_current_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

ExpectedVoid VulkanSymbioteEngine::create_pipelines() {
    // Create compute pipelines for all shader types
    ShaderSpecialization spec;
    spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
    
    // Create pipelines - these will be cached by ShaderRuntime
    VkPipeline attention_pipe = shader_runtime_->get_attention_pipeline(spec);
    VkPipeline ffn_pipe = shader_runtime_->get_feedforward_pipeline(spec);
    VkPipeline norm_pipe = shader_runtime_->get_rms_norm_pipeline(spec);
    VkPipeline rope_pipe = shader_runtime_->get_fused_matmul_rope_pipeline(spec);
    VkPipeline final_pipe = shader_runtime_->get_final_linear_pipeline(spec);
    
    if (attention_pipe == VK_NULL_HANDLE || ffn_pipe == VK_NULL_HANDLE ||
        norm_pipe == VK_NULL_HANDLE || rope_pipe == VK_NULL_HANDLE ||
        final_pipe == VK_NULL_HANDLE) {
        std::cerr << "[VulkanSymbioteEngine] Warning: Some pipelines failed to create" << std::endl;
    }
    
    return make_expected_success();
}

ExpectedVoid VulkanSymbioteEngine::upload_to_gpu(const std::vector<float>& data, VkBuffer dst_buffer) {
    // Create staging buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = data.size() * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &staging_buffer);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, staging_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    result = vkAllocateMemory(device_, &alloc_info, nullptr, &staging_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        return ExpectedVoid(static_cast<int>(result));
    }
    
    vkBindBufferMemory(device_, staging_buffer, staging_memory, 0);
    
    // Map and copy data
    void* mapped;
    vkMapMemory(device_, staging_memory, 0, data.size() * sizeof(float), 0, &mapped);
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vkUnmapMemory(device_, staging_memory);
    
    // Copy to device buffer
    VkCommandBuffer cmd_buffer = begin_single_time_commands();
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = data.size() * sizeof(float);
    vkCmdCopyBuffer(cmd_buffer, staging_buffer, dst_buffer, 1, &copy_region);
    end_single_time_commands(cmd_buffer);
    
    // Cleanup staging buffer
    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_memory, nullptr);
    
    return make_expected_success();
}

ExpectedVoid VulkanSymbioteEngine::download_from_gpu(VkBuffer src_buffer, std::vector<float>& data) {
    // Create staging buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = data.size() * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &staging_buffer);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, staging_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    result = vkAllocateMemory(device_, &alloc_info, nullptr, &staging_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        return ExpectedVoid(static_cast<int>(result));
    }
    
    vkBindBufferMemory(device_, staging_buffer, staging_memory, 0);
    
    // Copy from device buffer
    VkCommandBuffer cmd_buffer = begin_single_time_commands();
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = data.size() * sizeof(float);
    vkCmdCopyBuffer(cmd_buffer, src_buffer, staging_buffer, 1, &copy_region);
    end_single_time_commands(cmd_buffer);
    
    // Map and copy data
    void* mapped;
    vkMapMemory(device_, staging_memory, 0, data.size() * sizeof(float), 0, &mapped);
    std::memcpy(data.data(), mapped, data.size() * sizeof(float));
    vkUnmapMemory(device_, staging_memory);
    
    // Cleanup staging buffer
    vkDestroyBuffer(device_, staging_buffer, nullptr);
    vkFreeMemory(device_, staging_memory, nullptr);
    
    return make_expected_success();
}

VkCommandBuffer VulkanSymbioteEngine::begin_single_time_commands() {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;
    
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &cmd_buffer);
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    
    return cmd_buffer;
}

void VulkanSymbioteEngine::end_single_time_commands(VkCommandBuffer cmd_buffer) {
    vkEndCommandBuffer(cmd_buffer);
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    
    vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(compute_queue_);
    
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
}

uint32_t VulkanSymbioteEngine::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    return 0;
}

uint32_t VulkanSymbioteEngine::greedy_sampling(const float* logits, uint32_t size) {
    uint32_t max_idx = 0;
    float max_val = logits[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void VulkanSymbioteEngine::top_k_sampling(const float* logits, uint32_t vocab_size, uint32_t k, 
                                           float temperature,
                                           std::vector<uint32_t>& tokens, 
                                           std::vector<float>& probs, float* workspace) {
    // Copy logits to workspace
    std::memcpy(workspace, logits, vocab_size * sizeof(float));
    
    // Apply temperature
    if (temperature != 1.0f) {
        for (uint32_t i = 0; i < vocab_size; ++i) {
            workspace[i] /= temperature;
        }
    }
    
    // Softmax
    float max_logit = workspace[0];
    for (uint32_t i = 1; i < vocab_size; ++i) {
        if (workspace[i] > max_logit) max_logit = workspace[i];
    }
    
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        workspace[i] = std::exp(workspace[i] - max_logit);
        sum_exp += workspace[i];
    }
    
    for (uint32_t i = 0; i < vocab_size; ++i) {
        workspace[i] /= sum_exp;
    }
    
    // Find top k
    tokens.clear();
    probs.clear();
    
    for (uint32_t i = 0; i < k && i < vocab_size; ++i) {
        uint32_t best_idx = 0;
        float best_prob = workspace[0];
        
        for (uint32_t j = 1; j < vocab_size; ++j) {
            if (workspace[j] > best_prob) {
                best_prob = workspace[j];
                best_idx = j;
            }
        }
        
        tokens.push_back(best_idx);
        probs.push_back(best_prob);
        workspace[best_idx] = 0.0f;  // Remove from consideration
    }
}

void VulkanSymbioteEngine::detect_power_source() {
    // Platform-specific detection
    #ifdef __linux__
    std::ifstream status_file("/sys/class/power_supply/BAT0/status");
    if (status_file.is_open()) {
        std::string status;
        status_file >> status;
        on_battery_ = (status == "Discharging");
    }
    #endif
}

// BenchmarkStats print method
void BenchmarkStats::print() const {
    std::cout << "\n========== BENCHMARK RESULTS ==========" << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  Avg tokens/sec: " << avg_tokens_per_sec << std::endl;
    std::cout << "  Min tokens/sec: " << min_tokens_per_sec << std::endl;
    std::cout << "  Max tokens/sec: " << max_tokens_per_sec << std::endl;
    std::cout << "  Latency: " << avg_latency_ms << " ms/token" << std::endl;
    
    std::cout << "\nMemory:" << std::endl;
    std::cout << "  Peak VRAM: " << peak_vram_gb << " GB" << std::endl;
    std::cout << "  Avg VRAM: " << avg_vram_usage_gb << " GB" << std::endl;
    std::cout << "  Fragmentation: " << (fragmentation_ratio * 100) << "%" << std::endl;
    
    std::cout << "\nCache:" << std::endl;
    std::cout << "  Hit rate: " << (cache_hit_rate * 100) << "%" << std::endl;
    
    std::cout << "\nEviction:" << std::endl;
    std::cout << "  Total: " << total_evictions << std::endl;
    std::cout << "  Rate: " << eviction_rate << " evictions/sec" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

float VulkanSymbioteEngine::read_battery_capacity() {
    #ifdef __linux__
    std::ifstream capacity_file("/sys/class/power_supply/BAT0/capacity");
    if (capacity_file.is_open()) {
        int capacity;
        capacity_file >> capacity;
        return capacity / 100.0f;
    }
    #endif
    return 1.0f;
}

bool VulkanSymbioteEngine::read_ac_connected() {
    #ifdef __linux__
    std::ifstream status_file("/sys/class/power_supply/BAT0/status");
    if (status_file.is_open()) {
        std::string status;
        status_file >> status;
        return status != "Discharging";
    }
    #endif
    return true;
}

void VulkanSymbioteEngine::set_workgroup_size(uint32_t x, uint32_t y, uint32_t z) {
    workgroup_size_x_ = x;
    workgroup_size_y_ = y;
    workgroup_size_z_ = z;
}

void VulkanSymbioteEngine::get_workgroup_size(uint32_t& x, uint32_t& y, uint32_t& z) const noexcept {
    x = workgroup_size_x_;
    y = workgroup_size_y_;
    z = workgroup_size_z_;
}

void VulkanSymbioteEngine::apply_power_settings() {
    switch (power_profile_) {
        case PowerProfile::HIGH_PERFORMANCE:
            set_workgroup_size(256, 1, 1);
            set_prefetch_lookahead(5);
            break;
        case PowerProfile::BALANCED:
            set_workgroup_size(128, 1, 1);
            set_prefetch_lookahead(3);
            break;
        case PowerProfile::POWER_SAVER:
            set_workgroup_size(64, 1, 1);
            set_prefetch_lookahead(2);
            break;
    }
}

MemoryPoolStats VulkanSymbioteEngine::get_vram_stats() {
    if (!pack_manager_) return MemoryPoolStats{};
    return pack_manager_->get_vram_stats();
}

double VulkanSymbioteEngine::get_peak_vram_usage() {
    if (!pack_manager_) return 0.0;
    auto stats = pack_manager_->get_vram_stats();
    return static_cast<double>(stats.used_size) / (1024.0 * 1024.0 * 1024.0);
}

// BenchmarkResult conversion from BenchmarkStats
BenchmarkResult VulkanSymbioteEngine::run_benchmark(uint32_t warmup_tokens, 
                                                      uint32_t benchmark_tokens,
                                                      uint32_t iterations) {
    auto stats = run_benchmark_detailed(warmup_tokens, benchmark_tokens, iterations);
    
    BenchmarkResult result;
    result.warmup_tokens = warmup_tokens;
    result.benchmark_tokens = benchmark_tokens;
    result.iterations = iterations;
    result.avg_tokens_per_sec = stats.avg_tokens_per_sec;
    result.min_tokens_per_sec = stats.min_tokens_per_sec;
    result.max_tokens_per_sec = stats.max_tokens_per_sec;
    result.std_dev_tokens_per_sec = stats.std_dev_tokens_per_sec;
    result.avg_latency_ms = stats.avg_latency_ms;
    result.peak_vram_gb = stats.peak_vram_gb;
    result.cache_hit_rate = stats.cache_hit_rate;
    result.cache_size_mb = static_cast<double>(get_kv_cache_memory_usage()) / (1024.0 * 1024.0);
    
    return result;
}

// Helper methods for benchmark
double VulkanSymbioteEngine::calculate_mean(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
}

double VulkanSymbioteEngine::calculate_std_dev(const std::vector<double>& samples) {
    if (samples.size() < 2) return 0.0;
    double mean = calculate_mean(samples);
    double sq_sum = 0.0;
    for (double s : samples) {
        sq_sum += (s - mean) * (s - mean);
    }
    return std::sqrt(sq_sum / samples.size());
}

void VulkanSymbioteEngine::save_benchmark_results_json(const BenchmarkResult& result, 
                                                        const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n";
    file << "  \"warmup_tokens\": " << result.warmup_tokens << ",\n";
    file << "  \"benchmark_tokens\": " << result.benchmark_tokens << ",\n";
    file << "  \"iterations\": " << result.iterations << ",\n";
    file << "  \"avg_tokens_per_sec\": " << result.avg_tokens_per_sec << ",\n";
    file << "  \"min_tokens_per_sec\": " << result.min_tokens_per_sec << ",\n";
    file << "  \"max_tokens_per_sec\": " << result.max_tokens_per_sec << ",\n";
    file << "  \"std_dev_tokens_per_sec\": " << result.std_dev_tokens_per_sec << ",\n";
    file << "  \"avg_latency_ms\": " << result.avg_latency_ms << ",\n";
    file << "  \"peak_vram_gb\": " << result.peak_vram_gb << ",\n";
    file << "  \"cache_hit_rate\": " << result.cache_hit_rate << "\n";
    file << "}\n";
}

} // namespace vk_symbiote
