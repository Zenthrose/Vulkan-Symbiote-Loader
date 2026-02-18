#pragma once

#include "Common.h"
#include "VulkanSymbioteEngine.h"
#include <memory>
#include <vector>
#include <string>
#include <chrono>

namespace vk_symbiote {

// Forward declarations
class CooperativeMatrixManager;
class AutoTuner;

// Shader specialization constants
struct ShaderSpecialization {
    uint32_t workgroup_size_x = 16;
    uint32_t workgroup_size_y = 16;
    uint32_t workgroup_size_z = 1;
    bool use_subgroup_ops = true;
    uint32_t subgroup_size = 32;
    bool use_fp16_math = true;
};

// Tuning configuration
struct TuningConfig {
    uint32_t optimal_workgroup_size = 256;
    uint32_t optimal_subgroup_size = 32;
    bool use_cooperative_matrix = false;
    uint32_t coop_matrix_m = 16;
    uint32_t coop_matrix_n = 16;
    uint32_t coop_matrix_k = 16;
    bool prefer_shared_memory = true;
    uint32_t shared_memory_size = 16384;
    bool use_fp16 = true;
    uint32_t vendor_id = 0;
    std::string device_name;
    uint32_t matmul_workgroup_x = 16;
    uint32_t matmul_workgroup_y = 16;
    uint32_t attention_workgroup = 64;
    uint32_t reduction_workgroup = 256;
    
    void save(const std::string& path) const;
    bool load(const std::string& path);
};

// ShaderBenchmark class definition (must be complete for unique_ptr)
class ShaderBenchmark {
public:
    struct BenchmarkResult {
        uint32_t workgroup_size;
        double avg_time_ms;
        double std_dev_ms;
        double throughput_gflops;
        bool valid;
    };

    ShaderBenchmark(VkDevice device, VkPhysicalDevice physical_device, 
                    VkQueue queue, VkCommandPool command_pool);
    ~ShaderBenchmark();

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue queue_;
    VkCommandPool command_pool_;
    
    void create_benchmark_resources();
    void cleanup_resources();
};

class ShaderRuntime {
public:
    struct DeviceCapabilities {
        uint32_t max_compute_workgroup_invocations = 1024;
        uint32_t max_compute_workgroup_size[3] = {1024, 1024, 1024};
        uint32_t subgroup_size = 32;
        bool supports_subgroup_arithmetic = true;
        bool supports_fp16 = true;
        bool supports_int8 = false;
        uint32_t max_push_constant_size = 128;
        
        // Cooperative matrix support
        bool supports_cooperative_matrix = false;
        uint32_t cooperative_matrix_m = 16;
        uint32_t cooperative_matrix_n = 16;
        uint32_t cooperative_matrix_k = 16;
        
        // Device-specific tuning
        uint32_t optimal_workgroup_size = 256;
        uint32_t wave_size = 32;
        bool prefers_warp_shuffle = false;
        
        // Shared memory
        uint32_t shared_memory_size = 16384;
    };
    
    explicit ShaderRuntime(VkDevice device, VkPhysicalDevice physical_device, 
                          VkQueue compute_queue, VkCommandPool command_pool, 
                          VkDescriptorPool descriptor_pool);
    ~ShaderRuntime();
    
    // Pipeline creation and management
    VkPipeline create_specialized_pipeline(VkShaderModule shader_module,
                                           const ShaderSpecialization& spec);
    
    // Built-in shader templates
    VkPipeline get_fused_matmul_rope_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_attention_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_feedforward_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_rms_norm_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_final_linear_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_embedding_lookup_pipeline(const ShaderSpecialization& spec);
    
    // Shader compilation
    Expected<VkShaderModule> compile_compute_shader(const std::string& glsl_source,
                                                    const std::vector<const char*>& defines = {});
    
    // Descriptor set management
    VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
    void update_descriptor_set(VkDescriptorSet descriptor_set,
                               const std::vector<VkDescriptorBufferInfo>& buffer_infos);
    
    // Pipeline cache and query
    DeviceCapabilities get_device_capabilities() const { return device_caps_; }
    VkPipelineCache get_pipeline_cache() const { return pipeline_cache_; }
    
    // Pipeline cache persistence
    void load_pipeline_cache();
    void save_pipeline_cache() const;
    static std::string get_cache_directory();
    static std::string get_cache_file_path();
    
    // Layout accessors for descriptor set management
    VkDescriptorSetLayout get_descriptor_set_layout() const { return descriptor_set_layout_; }
    VkPipelineLayout get_pipeline_layout() const { return pipeline_layout_; }
    
    // Compute dispatch
    void dispatch_compute(VkPipeline pipeline,
                          VkDescriptorSet descriptor_set,
                          uint32_t group_count_x,
                          uint32_t group_count_y = 1,
                          uint32_t group_count_z = 1);
    
    // Auto-tuning: Optimize shader parameters for this device
    void auto_tune_shaders();
    ShaderSpecialization get_optimal_specialization(uint32_t operation_type) const;
    
    // Cooperative matrix support
    bool has_cooperative_matrix() const { return device_caps_.supports_cooperative_matrix; }
    VkPipeline get_cooperative_matmul_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_tuned_matmul_pipeline(uint32_t m, uint32_t n, uint32_t k);
    
    // Device-specific shader generation
    std::string generate_tuned_matmul_shader(uint32_t m, uint32_t n, uint32_t k);
    std::string generate_tuned_attention_shader(uint32_t seq_len, uint32_t head_dim);
    std::string generate_cooperative_matmul_shader_fp32();
    VkPipeline create_cooperative_matmul_pipeline();

private:
    // Vendor-specific tuning functions
    void tune_for_nvidia(const VkPhysicalDeviceProperties& props);
    void tune_for_amd(const VkPhysicalDeviceProperties& props);
    void tune_for_intel(const VkPhysicalDeviceProperties& props);
    void tune_for_arm(const VkPhysicalDeviceProperties& props);
    void tune_generic(const VkPhysicalDeviceProperties& props);
    bool check_cooperative_matrix_support();
    
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue compute_queue_;
    VkCommandPool command_pool_;
    VkDescriptorPool descriptor_pool_;
    DeviceCapabilities device_caps_;
    VkPipelineCache pipeline_cache_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipelineLayout pipeline_layout_;
    
    // Shader cache
    std::vector<VkShaderModule> shader_modules_;
    
    // Named pipeline cache (shader name -> pipeline)
    std::unordered_map<std::string, VkPipeline> named_pipeline_cache_;
    
    // Cooperative matrix manager
    std::unique_ptr<CooperativeMatrixManager> coop_matrix_mgr_;
    
    // Auto-tuner
    std::unique_ptr<AutoTuner> auto_tuner_;
    
    // Benchmark system
    std::unique_ptr<ShaderBenchmark> benchmark_;
    
    // Private methods
    DeviceCapabilities query_device_capabilities() const;
    VkDescriptorSetLayout create_descriptor_set_layout() const;
    VkPipelineLayout create_pipeline_layout() const;
    
    // Shader compilation helpers
    Expected<VkShaderModule> compile_glsl_to_spirv(const std::string& glsl_source,
                                                   const std::vector<const char*>& defines);
    
    // Pipeline creation helpers
    VkComputePipelineCreateInfo create_compute_pipeline_info(VkShaderModule shader_module,
                                                             VkPipelineLayout layout,
                                                             const VkSpecializationInfo* specialization_info) const;
    
    // Specialization data creation
    VkSpecializationInfo create_specialization_info(const ShaderSpecialization& spec,
                                                    std::vector<uint32_t>& specialization_data) const;
    
    // Time helpers
    static uint64_t get_current_time_ns();
};

} // namespace vk_symbiote
