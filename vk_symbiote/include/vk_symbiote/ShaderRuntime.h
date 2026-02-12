#pragma once

#include "Common.h"
#include "VulkanSymbioteEngine.h"

namespace vk_symbiote {

class ShaderRuntime {
public:
    struct ShaderSpecialization {
        uint32_t workgroup_size_x = 16;
        uint32_t workgroup_size_y = 16;
        uint32_t workgroup_size_z = 1;
        bool use_subgroup_ops = true;
        uint32_t subgroup_size = 32;
        bool use_fp16_math = true;
    };
    
    struct DeviceCapabilities {
        uint32_t max_compute_workgroup_invocations = 1024;
        uint32_t max_compute_workgroup_size[3] = {1024, 1024, 1024};
        uint32_t subgroup_size = 32;
        bool supports_subgroup_arithmetic = true;
        bool supports_fp16 = true;
        bool supports_int8 = false;
        uint32_t max_push_constant_size = 128;
    };
    
    explicit ShaderRuntime(VkDevice device, VkPhysicalDevice physical_device, VkQueue compute_queue, VkCommandPool command_pool, VkDescriptorPool descriptor_pool);
    ~ShaderRuntime();
    
    // Pipeline creation and management
    VkPipeline create_specialized_pipeline(
        VkShaderModule shader_module,
        const ShaderSpecialization& spec);
    
    // Built-in shader templates
    VkPipeline get_fused_matmul_rope_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_attention_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_feedforward_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_rms_norm_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_final_linear_pipeline(const ShaderSpecialization& spec);
    
    // Shader compilation
    Expected<VkShaderModule> compile_compute_shader(
        const std::string& glsl_source,
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

private:
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
    
    // Private methods
    DeviceCapabilities query_device_capabilities() const;
    VkDescriptorSetLayout create_descriptor_set_layout() const;
    VkPipelineLayout create_pipeline_layout() const;
    
    // Shader compilation helpers
    Expected<VkShaderModule> compile_glsl_to_spirv(
        const std::string& glsl_source,
        const std::vector<const char*>& defines);
    
    // Pipeline creation helpers
    VkComputePipelineCreateInfo create_compute_pipeline_info(
        VkShaderModule shader_module,
        VkPipelineLayout layout,
        const VkSpecializationInfo* specialization_info) const;
    
    // Specialization data creation
    VkSpecializationInfo create_specialization_info(
        const ShaderSpecialization& spec,
        std::vector<uint32_t>& specialization_data) const;
};

} // namespace vk_symbiote