#pragma once

#include "Common.h"
#include "NomadPack.h"
#include "VitalityOracle.h"
#include "GGUFLoader.h"
#include <vulkan/vulkan.h>

namespace vk_symbiote {

class ShaderRuntime;
class Tokenizer;

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

    std::vector<float> hidden_states_;
    std::vector<uint32_t> token_sequence_;
    uint32_t current_position_ = 0;
    
    // Performance monitoring
    PerformanceMetrics performance_metrics_;

    ExpectedVoid initialize_vulkan();
    ExpectedVoid load_model();
    ExpectedVoid create_pipelines();
    ExpectedVoid destroy_resources();

    Expected<std::vector<float>> embed_tokens(const std::vector<uint32_t>& tokens);
    Expected<std::vector<float>> forward_layer(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> attention(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> feed_forward(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> rms_norm(const std::vector<float>& hidden, uint32_t layer_idx);
    Expected<std::vector<float>> apply_rope(const std::vector<float>& hidden, uint32_t position);
    Expected<std::vector<float>> final_projection(const std::vector<float>& hidden);

    void schedule_prefetch(uint32_t current_layer, uint32_t lookahead);
    void evict_low_priority();

    ExpectedVoid upload_to_gpu(const std::vector<float>& data, VkBuffer buffer);
    ExpectedVoid download_from_gpu(VkBuffer buffer, std::vector<float>& data);
    
    // Performance monitoring
    const PerformanceMetrics& get_performance_metrics() const { return performance_metrics_; }
    void reset_performance_metrics() { performance_metrics_.reset(); }
};

} // namespace vk_symbiote
