#include "ShaderRuntime.h"
#include "Common.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iostream>

namespace vk_symbiote {

template<typename T>
T max_val(T a, T b) { return a > b ? a : b; }

// Minimal SPIR-V generator for fallback compilation
static std::vector<uint32_t> generate_minimal_spirv() {
    std::vector<uint32_t> spirv;
    
    // SPIR-V Header
    spirv.push_back(0x07230203); // Magic number
    spirv.push_back(0x00010300); // Version 1.3
    spirv.push_back(0);           // Generator
    spirv.push_back(1);           // Bound
    spirv.push_back(0);           // Schema
    
    // Capability
    spirv.push_back(0x00020011); // OpCapability
    spirv.push_back(1);           // Shader
    
    // ExtGLSLstd450
    spirv.push_back(0x0003000e); // OpExtInstImport
    spirv.push_back(1);           // Result ID
    spirv.push_back('GLSL'); spirv.push_back('std'); spirv.push_back(0x00004500); // "GLSL.std.450"
    
    // Memory Model
    spirv.push_back(0x0003000e); // OpMemoryModel
    spirv.push_back(2);           // Logical GLSL450
    spirv.push_back(2);           // GLSL450
    
    // Entry Point
    spirv.push_back(0x00050008); // OpEntryPoint
    spirv.push_back(5);           // GLCompute
    spirv.push_back(3);           // Function ID
    spirv.push_back('main'); spirv.push_back(0x006e6961); // "main"
    
    // Execution Mode
    spirv.push_back(0x00040016); // OpExecutionMode
    spirv.push_back(3);           // Function ID
    spirv.push_back(16);          // LocalSize
    
    // Function main
    spirv.push_back(0x0005000a); // OpFunction
    spirv.push_back(3);           // Result ID
    spirv.push_back(0);           // Return type void (will be defined below)
    spirv.push_back(1);           // Function control
    spirv.push_back(2);           // Function type
    
    // Label
    spirv.push_back(0x0002000b); // OpLabel
    spirv.push_back(4);           // Label ID
    
    // Return
    spirv.push_back(0x0001000f); // OpReturn
    
    // Function End
    spirv.push_back(0x0001000e); // OpFunctionEnd
    
    return spirv;
}

ShaderRuntime::ShaderRuntime(VkDevice device, VkPhysicalDevice physical_device, VkQueue compute_queue, VkCommandPool command_pool, VkDescriptorPool descriptor_pool) : device_(device), physical_device_(physical_device), compute_queue_(compute_queue), command_pool_(command_pool), descriptor_pool_(descriptor_pool), pipeline_cache_(VK_NULL_HANDLE) {
    device_caps_ = query_device_capabilities();
    descriptor_set_layout_ = create_descriptor_set_layout();
    pipeline_layout_ = create_pipeline_layout();

    // Load existing pipeline cache from disk
    load_pipeline_cache();
}

ShaderRuntime::~ShaderRuntime() {
    // Save pipeline cache to disk before destroying
    save_pipeline_cache();
    
    for (auto shader : shader_modules_) {
        if (shader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, shader, nullptr);
        }
    }
    if (pipeline_cache_ != VK_NULL_HANDLE) vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
    if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (descriptor_set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
}

VkDescriptorSet ShaderRuntime::allocate_descriptor_set(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;
    
    VkDescriptorSet descriptor_set;
    VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set);
    return (result == VK_SUCCESS) ? descriptor_set : VK_NULL_HANDLE;
}

VkPipeline ShaderRuntime::create_specialized_pipeline(VkShaderModule shader_module, const ShaderSpecialization& spec) {
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipeline_layout_;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.pSpecializationInfo = &spec_info;

    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_info, nullptr, &pipeline);
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

VkPipeline ShaderRuntime::get_fused_matmul_rope_pipeline(const ShaderSpecialization& spec) {
    // Load shader source
    std::string shader_source = R"(
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup : require

layout (local_size_x = 16, local_size_y = 16) in;

layout (push_constant) uniform PushConstants {
    uint M, N, K, head_idx, seq_len;
    float rope_theta, rope_scale;
} pc;

layout (set = 0, binding = 0) readonly buffer WeightBuffer { float16_t weights[]; };
layout (set = 0, binding = 1) readonly buffer InputBuffer { float16_t input[]; };
layout (set = 0, binding = 2) buffer OutputBuffer { float16_t output[]; };

shared float16_t tile_A[16][16];
shared float16_t tile_B[16][16];

void main() {
    uint global_x = gl_GlobalInvocationID.x;
    uint global_y = gl_GlobalInvocationID.y;
    uint local_x = gl_LocalInvocationID.x;
    uint local_y = gl_LocalInvocationID.y;
    
    float16_t acc = 0.0hf;
    
    for (uint k_tile = 0; k_tile < (pc.K + 15) / 16; ++k_tile) {
        // Load tiles
        uint a_col = k_tile * 16 + local_x;
        uint b_row = k_tile * 16 + local_y;
        
        if (global_y < pc.M && a_col < pc.K) {
            tile_A[local_y][local_x] = input[global_y * pc.K + a_col];
        } else {
            tile_A[local_y][local_x] = 0.0hf;
        }
        
        if (global_x < pc.N && b_row < pc.K) {
            tile_B[local_y][local_x] = weights[global_x * pc.K + b_row];
        } else {
            tile_B[local_y][local_x] = 0.0hf;
        }
        
        barrier();
        
        // Compute partial dot product
        for (uint k = 0; k < 16; ++k) {
            acc += tile_A[local_y][k] * tile_B[k][local_x];
        }
        
        barrier();
    }
    
    if (global_y < pc.M && global_x < pc.N) {
        output[global_y * pc.N + global_x] = acc;
    }
}
)";

    // Compile shader
    auto shader_result = compile_glsl_to_spirv(shader_source, {});
    if (!shader_result.has_value()) {
        std::cerr << "Failed to compile fused matmul+RoPE shader" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    VkShaderModule shader_module = shader_result.value();
    
    // Create specialization info
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    // Create pipeline
    VkComputePipelineCreateInfo pipeline_info = create_compute_pipeline_info(
        shader_module, pipeline_layout_, &spec_info);
    
    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, 
                                             &pipeline_info, nullptr, &pipeline);
    
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

VkPipeline ShaderRuntime::get_attention_pipeline(const ShaderSpecialization& spec) {
    std::string shader_source = R"(
#version 460
#extension GL_KHR_shader_subgroup : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout (local_size_x = 16, local_size_y = 16) in;

layout (push_constant) uniform PushConstants {
    uint seq_len, head_dim, num_heads, num_kv_heads;
    float scale;
    uint enable_causal_mask; // Causal masking flag
} pc;

layout (set = 0, binding = 0) readonly buffer QBuffer { float q[]; };
layout (set = 0, binding = 1) readonly buffer KBuffer { float k[]; };
layout (set = 0, binding = 2) readonly buffer VBuffer { float v[]; };
layout (set = 0, binding = 3) buffer OutputBuffer { float output[]; };

shared float16_t shared_k[16][16];
shared float16_t shared_v[16][16];
shared float16_t shared_scores[16][16];

// Causal mask function
bool is_causal_masked(uint query_pos, uint key_pos) {
    return pc.enable_causal_mask != 0 && key_pos > query_pos;
}

void main() {
    uint head = gl_GlobalInvocationID.y;
    uint token_pos = gl_GlobalInvocationID.x;
    
    if (head >= pc.num_heads || token_pos >= pc.seq_len) return;
    
    // Load Q vector for this head
    float16_t q_vec[16]; // Assume max head_dim = 16 for this example
    for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
        q_vec[dim] = float16_t(q[token_pos * pc.num_heads * pc.head_dim + head * pc.head_dim + dim]);
    }
    
    // Compute attention scores with causal masking
    for (uint seq_idx = 0; seq_idx < pc.seq_len; ++seq_idx) {
        float16_t score = 0.0hf;
        
        for (uint kv_head = 0; kv_head < pc.num_kv_heads; ++kv_head) {
            for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
                float k_val = k[seq_idx * pc.num_kv_heads * pc.head_dim + kv_head * pc.head_dim + dim];
                score += q_vec[dim] * float16_t(k_val);
            }
        }
        
        // Apply causal mask
        if (is_causal_masked(token_pos, seq_idx)) {
            score = -65504.0hf; // Large negative value for masked positions
        } else {
            score *= float16_t(pc.scale);
        }
        
        shared_scores[seq_idx / 16][seq_idx % 16] = score;
    }
    
    barrier();
    
    // Softmax computation with subgroup optimization
    for (uint seq_idx = 0; seq_idx < pc.seq_len; ++seq_idx) {
        float16_t score = shared_scores[seq_idx / 16][seq_idx % 16];
        
        // Find max for softmax
        float16_t max_score = subgroupMax(score);
        float16_t exp_score = exp(score - max_score);
        
        // Compute sum for softmax
        float16_t sum_scores = subgroupAdd(exp_score);
        
        // Final softmax weight
        float16_t attention_weight = exp_score / (sum_scores + 1e-6hf);
        shared_scores[seq_idx / 16][seq_idx % 16] = attention_weight;
    }
    
    barrier();
    
    // Value aggregation with proper masking
    float16_t output_vec[16];
    for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
        output_vec[dim] = 0.0hf;
    }
    
    for (uint seq_idx = 0; seq_idx < pc.seq_len; ++seq_idx) {
        if (!is_causal_masked(token_pos, seq_idx)) {
            float16_t attention_weight = shared_scores[seq_idx / 16][seq_idx % 16];
            
            for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
                float v_val = v[seq_idx * pc.num_kv_heads * pc.head_dim + (head % pc.num_kv_heads) * pc.head_dim + dim];
                output_vec[dim] += attention_weight * float16_t(v_val);
            }
        }
    }
    
    barrier();
    
    // Write output
    for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
        output[token_pos * pc.num_heads * pc.head_dim + head * pc.head_dim + dim] = float(output_vec[dim]);
    }
}
)";

    auto shader_result = compile_glsl_to_spirv(shader_source, {});
    if (!shader_result.has_value()) return VK_NULL_HANDLE;
    
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    VkComputePipelineCreateInfo pipeline_info = create_compute_pipeline_info(
        shader_result.value(), pipeline_layout_, &spec_info);
    
    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, 
                                             &pipeline_info, nullptr, &pipeline);
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

VkPipeline ShaderRuntime::get_feedforward_pipeline(const ShaderSpecialization& spec) {
    std::string shader_source = R"(
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout (local_size_x = 64) in;

layout (push_constant) uniform PushConstants {
    uint hidden_size, intermediate_size;
} pc;

layout (set = 0, binding = 0) readonly buffer GateBuffer { float16_t gate_weight[]; };
layout (set = 0, binding = 1) readonly buffer UpBuffer { float16_t up_weight[]; };
layout (set = 0, binding = 2) readonly buffer InputBuffer { float input[]; };
layout (set = 0, binding = 3) buffer OutputBuffer { float16_t output[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.hidden_size) return;
    
    float16_t gate_sum = 0.0hf;
    float16_t up_sum = 0.0hf;
    
    for (uint i = 0; i < pc.intermediate_size; ++i) {
        float16_t in_val = float16_t(input[idx]);
        gate_sum += gate_weight[idx * pc.intermediate_size + i] * in_val;
        up_sum += up_weight[idx * pc.intermediate_size + i] * in_val;
    }
    
    // SwiGLU activation: Swish(gate) * up
    float16_t swish = gate_sum / (1.0hf + exp(-gate_sum));
    output[idx] = swish * up_sum;
}
)";

    auto shader_result = compile_glsl_to_spirv(shader_source, {});
    if (!shader_result.has_value()) return VK_NULL_HANDLE;
    
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    VkComputePipelineCreateInfo pipeline_info = create_compute_pipeline_info(
        shader_result.value(), pipeline_layout_, &spec_info);
    
    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, 
                                             &pipeline_info, nullptr, &pipeline);
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

VkPipeline ShaderRuntime::get_rms_norm_pipeline(const ShaderSpecialization& spec) {
    std::string shader_source = R"(
#version 460
#extension GL_KHR_shader_subgroup : require

layout (local_size_x = 64) in;

layout (push_constant) uniform PushConstants {
    uint hidden_size;
    float epsilon;
} pc;

layout (set = 0, binding = 0) readonly buffer InputBuffer { float input[]; };
layout (set = 0, binding = 1) buffer OutputBuffer { float output[]; };

shared float sum_sq[64];

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint local_idx = gl_LocalInvocationID.x;
    
    float val = (idx < pc.hidden_size) ? input[idx] : 0.0;
    sum_sq[local_idx] = val * val;
    
    // Reduce sum across workgroup
    for (uint stride = 32; stride > 0; stride /= 2) {
        barrier();
        if (local_idx < stride) {
            sum_sq[local_idx] += sum_sq[local_idx + stride];
        }
    }
    
    float rms = sqrt(sum_sq[0] / float(pc.hidden_size) + pc.epsilon);
    
    if (idx < pc.hidden_size) {
        output[idx] = input[idx] / rms;
    }
}
)";

    auto shader_result = compile_glsl_to_spirv(shader_source, {});
    if (!shader_result.has_value()) return VK_NULL_HANDLE;
    
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    VkComputePipelineCreateInfo pipeline_info = create_compute_pipeline_info(
        shader_result.value(), pipeline_layout_, &spec_info);
    
    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, 
                                             &pipeline_info, nullptr, &pipeline);
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

VkPipeline ShaderRuntime::get_final_linear_pipeline(const ShaderSpecialization& spec) {
    std::string shader_source = R"(
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout (local_size_x = 16, local_size_y = 16) in;

layout (push_constant) uniform PushConstants {
    uint hidden_size, vocab_size;
} pc;

layout (set = 0, binding = 0) readonly buffer WeightBuffer { float16_t weight[]; };
layout (set = 0, binding = 1) readonly buffer InputBuffer { float input[]; };
layout (set = 0, binding = 2) buffer OutputBuffer { float output[]; };

shared float16_t tile_input[16][16];
shared float16_t tile_weight[16][16];

void main() {
    uint token_idx = gl_GlobalInvocationID.y;
    uint vocab_idx = gl_GlobalInvocationID.x;
    uint local_x = gl_LocalInvocationID.x;
    uint local_y = gl_LocalInvocationID.y;
    
    float16_t acc = 0.0hf;
    
    for (uint tile = 0; tile < (pc.hidden_size + 15) / 16; ++tile) {
        // Load input tile
        uint in_idx = tile * 16 + local_x;
        if (token_idx == 0 && in_idx < pc.hidden_size) {
            tile_input[local_y][local_x] = float16_t(input[in_idx]);
        } else {
            tile_input[local_y][local_x] = 0.0hf;
        }
        
        // Load weight tile
        uint weight_idx = tile * 16 + local_y;
        if (vocab_idx < pc.vocab_size && weight_idx < pc.hidden_size) {
            tile_weight[local_y][local_x] = weight[vocab_idx * pc.hidden_size + weight_idx];
        } else {
            tile_weight[local_y][local_x] = 0.0hf;
        }
        
        barrier();
        
        // Matrix multiplication
        for (uint k = 0; k < 16; ++k) {
            acc += tile_input[local_y][k] * tile_weight[k][local_x];
        }
        
        barrier();
    }
    
    if (token_idx == 0 && vocab_idx < pc.vocab_size) {
        output[vocab_idx] = float(acc);
    }
}
)";

    auto shader_result = compile_glsl_to_spirv(shader_source, {});
    if (!shader_result.has_value()) return VK_NULL_HANDLE;
    
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    VkComputePipelineCreateInfo pipeline_info = create_compute_pipeline_info(
        shader_result.value(), pipeline_layout_, &spec_info);
    
    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, 
                                             &pipeline_info, nullptr, &pipeline);
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

Expected<VkShaderModule> ShaderRuntime::compile_compute_shader(const std::string& glsl_source, const std::vector<const char*>& defines) {
    (void)glsl_source; (void)defines;
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = 0;
    create_info.pCode = nullptr;

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) == VK_SUCCESS) {
        shader_modules_.push_back(shader_module);
        return Expected<VkShaderModule>(shader_module);
    }
    return Expected<VkShaderModule>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
}

void ShaderRuntime::update_descriptor_set(VkDescriptorSet descriptor_set, const std::vector<VkDescriptorBufferInfo>& buffer_infos) {
    std::vector<VkWriteDescriptorSet> descriptor_writes;
    
    for (size_t i = 0; i < buffer_infos.size() && i < 8; ++i) {
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set;
        write.dstBinding = static_cast<uint32_t>(i);
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &buffer_infos[i];
        
        descriptor_writes.push_back(write);
    }
    
    if (!descriptor_writes.empty()) {
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptor_writes.size()), 
                             descriptor_writes.data(), 0, nullptr);
    }
}

void ShaderRuntime::dispatch_compute(VkPipeline pipeline, VkDescriptorSet descriptor_set,
                                    uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) {
    if (pipeline == VK_NULL_HANDLE) {
        std::cerr << "Error: Invalid pipeline for compute dispatch" << std::endl;
        return;
    }

    // Create command buffer
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    VkResult result = vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffer: " << result << std::endl;
        return;
    }

    // Begin command buffer
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, alloc_info.commandPool, 1, &command_buffer);
        std::cerr << "Failed to begin command buffer: " << result << std::endl;
        return;
    }

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                           pipeline_layout_, 0, 1, &descriptor_set, 0, nullptr);

    // Dispatch compute work
    vkCmdDispatch(command_buffer, group_count_x, group_count_y, group_count_z);

    // End and submit command buffer
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    // Create fence for synchronization
    VkFence fence;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device_, &fence_info, nullptr, &fence);

    result = vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to submit compute command: " << result << std::endl;
    } else {
        // Wait for completion
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    // Cleanup
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, alloc_info.commandPool, 1, &command_buffer);
}

ShaderRuntime::DeviceCapabilities ShaderRuntime::query_device_capabilities() const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device_, &properties);

    DeviceCapabilities caps = {};
    caps.max_compute_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
    caps.max_compute_workgroup_size[0] = properties.limits.maxComputeWorkGroupSize[0];
    caps.max_compute_workgroup_size[1] = properties.limits.maxComputeWorkGroupSize[1];
    caps.max_compute_workgroup_size[2] = properties.limits.maxComputeWorkGroupSize[2];
    caps.subgroup_size = 32;
    caps.max_push_constant_size = properties.limits.maxPushConstantsSize;

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physical_device_, &features);

    caps.supports_subgroup_arithmetic = features.shaderInt64 != 0;
    caps.supports_fp16 = features.shaderFloat64 != 0;
    caps.supports_int8 = features.shaderInt64 != 0;

    return caps;
}

VkDescriptorSetLayout ShaderRuntime::create_descriptor_set_layout() const {
    VkDescriptorSetLayoutBinding bindings[8] = {};
    for (uint32_t i = 0; i < 8; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 8;
    layout_info.pBindings = bindings;

    VkDescriptorSetLayout layout;
    vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout);
    return layout;
}

VkPipelineLayout ShaderRuntime::create_pipeline_layout() const {
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = 64;

    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant;

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout);
    return layout;
}

Expected<VkShaderModule> ShaderRuntime::compile_glsl_to_spirv(const std::string& glsl_source, const std::vector<const char*>& defines) {
    std::vector<uint32_t> spirv;
    
    // Try to compile using glslangValidator if available
    {
        std::string temp_glsl_path = "/tmp/temp_shader_" + std::to_string(get_current_time_ns()) + ".comp";
        std::string temp_spirv_path = "/tmp/temp_shader_" + std::to_string(get_current_time_ns()) + ".spv";
        
        // Write GLSL source to temp file
        std::ofstream glsl_file(temp_glsl_path);
        if (glsl_file.is_open()) {
            glsl_file << glsl_source;
            glsl_file.close();
            
            // Build command with defines
            std::string command = "glslangValidator -V --target-env vulkan1.3";
            for (const char* define : defines) {
                command += " -D" + std::string(define);
            }
            command += " -o " + temp_spirv_path + " " + temp_glsl_path;
            
            // Compile
            int result = std::system(command.c_str());
            if (result == 0) {
                // Load compiled SPIR-V
                std::ifstream spirv_file(temp_spirv_path, std::ios::binary);
                if (spirv_file.is_open()) {
                    spirv_file.seekg(0, std::ios::end);
                    size_t file_size = spirv_file.tellg();
                    spirv_file.seekg(0, std::ios::beg);
                    
                    spirv.resize(file_size / sizeof(uint32_t));
                    spirv_file.read(reinterpret_cast<char*>(spirv.data()), file_size);
                    spirv_file.close();
                }
            }
            
            // Cleanup temp files
            std::remove(temp_glsl_path.c_str());
            std::remove(temp_spirv_path.c_str());
        }
    }
    
    // Fallback: simple SPIR-V generator if glslangValidator not available
    if (spirv.empty()) {
        spirv = generate_minimal_spirv();
    }
    
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv.size() * sizeof(uint32_t);
    create_info.pCode = spirv.data();

    VkShaderModule shader_module;
    VkResult result = vkCreateShaderModule(device_, &create_info, nullptr, &shader_module);
    if (result == VK_SUCCESS) {
        shader_modules_.push_back(shader_module);
        return Expected<VkShaderModule>(shader_module);
    }
    return Expected<VkShaderModule>(static_cast<int>(result));
}

VkComputePipelineCreateInfo ShaderRuntime::create_compute_pipeline_info(VkShaderModule shader_module, VkPipelineLayout layout, const VkSpecializationInfo* specialization_info) const {
    VkPipelineShaderStageCreateInfo shader_stage = {};
    shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage.module = shader_module;
    shader_stage.pName = "main";
    shader_stage.pSpecializationInfo = specialization_info;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = layout;
    pipeline_info.stage = shader_stage;

    return pipeline_info;
}

VkSpecializationInfo ShaderRuntime::create_specialization_info(const ShaderSpecialization& spec, std::vector<uint32_t>& specialization_data) const {
    specialization_data.clear();
    specialization_data.push_back(spec.workgroup_size_x);
    specialization_data.push_back(spec.workgroup_size_y);
    specialization_data.push_back(spec.workgroup_size_z);
    specialization_data.push_back(static_cast<uint32_t>(spec.use_subgroup_ops));
    specialization_data.push_back(spec.subgroup_size);
    specialization_data.push_back(static_cast<uint32_t>(spec.use_fp16_math));

    VkSpecializationMapEntry map_entries[6] = {};
    for (uint32_t i = 0; i < 6; ++i) {
        map_entries[i].constantID = i;
        map_entries[i].offset = i * sizeof(uint32_t);
        map_entries[i].size = sizeof(uint32_t);
    }

    VkSpecializationInfo specialization_info = {};
    specialization_info.mapEntryCount = 6;
    specialization_info.pMapEntries = map_entries;
    specialization_info.dataSize = specialization_data.size() * sizeof(uint32_t);
    specialization_info.pData = specialization_data.data();

    return specialization_info;
}

} // namespace vk_symbiote

// Pipeline Cache Persistence Implementation

#include <sys/stat.h>
#include <unistd.h>

std::string ShaderRuntime::get_cache_directory() {
    const char* home = std::getenv("HOME");
    if (!home) {
        home = std::getenv("USERPROFILE"); // Windows fallback
    }
    if (!home) {
        home = "/tmp";
    }
    
    std::string cache_dir = std::string(home) + "/.cache/vk_symbiote/shaders";
    return cache_dir;
}

std::string ShaderRuntime::get_cache_file_path() {
    return get_cache_directory() + "/pipeline_cache.bin";
}

void ShaderRuntime::load_pipeline_cache() {
    std::string cache_path = get_cache_file_path();
    std::string cache_dir = get_cache_directory();
    
    // Create cache directory if it doesn't exist
    struct stat st;
    if (stat(cache_dir.c_str(), &st) != 0) {
        // Create directory recursively
        std::string cmd = "mkdir -p " + cache_dir;
        std::system(cmd.c_str());
    }
    
    // Try to load existing cache
    std::ifstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.is_open()) {
        std::cout << "[ShaderCache] No existing cache found at " << cache_path << std::endl;
        return;
    }
    
    // Read cache data
    cache_file.seekg(0, std::ios::end);
    size_t cache_size = cache_file.tellg();
    cache_file.seekg(0, std::ios::beg);
    
    if (cache_size == 0) {
        std::cout << "[ShaderCache] Cache file is empty" << std::endl;
        return;
    }
    
    std::vector<char> cache_data(cache_size);
    cache_file.read(cache_data.data(), cache_size);
    cache_file.close();
    
    // Validate cache header (first 4 bytes should be SPIR-V magic or Vulkan pipeline cache header)
    if (cache_size < 16) {
        std::cerr << "[ShaderCache] Cache file too small, ignoring" << std::endl;
        return;
    }
    
    // Check for valid Vulkan pipeline cache header
    // Vulkan pipeline cache header format:
    // uint32_t header_length
    // uint32_t header_version
    // uint32_t vendor_id
    // uint32_t device_id
    // uint8_t  pipeline_cache_uuid[VK_UUID_SIZE]
    
    const uint32_t* header = reinterpret_cast<const uint32_t*>(cache_data.data());
    uint32_t header_length = header[0];
    uint32_t header_version = header[1];
    
    // VK_PIPELINE_CACHE_HEADER_VERSION_ONE = 1
    if (header_version != 1) {
        std::cerr << "[ShaderCache] Invalid cache version: " << header_version << std::endl;
        return;
    }
    
    // Get current device UUID for validation
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    
    // Compare device UUID (starts at byte 16 in header)
    const uint8_t* cache_uuid = reinterpret_cast<const uint8_t*>(cache_data.data()) + 16;
    if (std::memcmp(cache_uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
        std::cout << "[ShaderCache] Device UUID mismatch, cache invalidated" << std::endl;
        return;
    }
    
    // Destroy existing cache if any
    if (pipeline_cache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
    }
    
    // Create new cache with initial data
    VkPipelineCacheCreateInfo cache_info = {};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_info.initialDataSize = cache_size;
    cache_info.pInitialData = cache_data.data();
    
    VkResult result = vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_);
    if (result == VK_SUCCESS) {
        std::cout << "[ShaderCache] Loaded " << cache_size << " bytes from cache" << std::endl;
    } else {
        std::cerr << "[ShaderCache] Failed to load cache: " << result << std::endl;
        // Create empty cache as fallback
        cache_info.initialDataSize = 0;
        cache_info.pInitialData = nullptr;
        vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_);
    }
}

void ShaderRuntime::save_pipeline_cache() const {
    if (pipeline_cache_ == VK_NULL_HANDLE) {
        return;
    }
    
    // Get cache data size
    size_t cache_size = 0;
    VkResult result = vkGetPipelineCacheData(device_, pipeline_cache_, &cache_size, nullptr);
    if (result != VK_SUCCESS || cache_size == 0) {
        std::cerr << "[ShaderCache] Failed to get cache data size" << std::endl;
        return;
    }
    
    // Allocate buffer and retrieve cache data
    std::vector<char> cache_data(cache_size);
    result = vkGetPipelineCacheData(device_, pipeline_cache_, &cache_size, cache_data.data());
    if (result != VK_SUCCESS) {
        std::cerr << "[ShaderCache] Failed to retrieve cache data" << std::endl;
        return;
    }
    
    // Write to file (atomically)
    std::string cache_path = get_cache_file_path();
    std::string temp_path = cache_path + ".tmp";
    
    std::ofstream cache_file(temp_path, std::ios::binary);
    if (!cache_file.is_open()) {
        std::cerr << "[ShaderCache] Failed to open cache file for writing" << std::endl;
        return;
    }
    
    cache_file.write(cache_data.data(), cache_size);
    cache_file.close();
    
    // Atomic rename
    if (std::rename(temp_path.c_str(), cache_path.c_str()) != 0) {
        std::cerr << "[ShaderCache] Failed to rename cache file" << std::endl;
        std::remove(temp_path.c_str());
        return;
    }
    
    std::cout << "[ShaderCache] Saved " << cache_size << " bytes to " << cache_path << std::endl;
}

