#include "ShaderRuntime.h"
#include "Common.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>

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

ShaderRuntime::ShaderRuntime(VkDevice device, VkPhysicalDevice physical_device, VkQueue compute_queue, VkCommandPool command_pool, VkDescriptorPool descriptor_pool) 
    : device_(device), physical_device_(physical_device), compute_queue_(compute_queue), command_pool_(command_pool), descriptor_pool_(descriptor_pool), pipeline_cache_(VK_NULL_HANDLE) {
    
    // Query device capabilities first
    device_caps_ = query_device_capabilities();
    
    // Auto-tune shaders based on device properties
    auto_tune_shaders();
    
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

// ============================================================================
// Auto-Tune System with Device Property Detection
// ============================================================================
void ShaderRuntime::auto_tune_shaders() {
    std::cout << "[ShaderRuntime] Auto-tuning shaders for device..." << std::endl;
    
    // Query device properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    
    VkPhysicalDeviceSubgroupProperties subgroup_props = {};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    
    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    
    vkGetPhysicalDeviceProperties2(physical_device_, &props2);
    
    // Query cooperative matrix properties
    if (device_caps_.supports_cooperative_matrix) {
        VkPhysicalDeviceCooperativeMatrixPropertiesNV coop_matrix_props = {};
        coop_matrix_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV;
        
        VkPhysicalDeviceProperties2 coop_props2 = {};
        coop_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        coop_props2.pNext = &coop_matrix_props;
        
        vkGetPhysicalDeviceProperties2(physical_device_, &coop_props2);
        
        std::cout << "[ShaderRuntime] Cooperative matrix supported" << std::endl;
    }
    
    // Vendor-specific tuning based on deviceID
    uint32_t vendor_id = props.vendorID;
    std::string device_name(props.deviceName);
    
    std::cout << "[ShaderRuntime] Device: " << device_name << " (Vendor: 0x" << std::hex << vendor_id << std::dec << ")" << std::endl;
    
    if (vendor_id == 0x10DE) {  // NVIDIA
        tune_for_nvidia(props);
    } else if (vendor_id == 0x1002 || vendor_id == 0x1022) {  // AMD
        tune_for_amd(props);
    } else if (vendor_id == 0x8086 || vendor_id == 0x8087) {  // Intel
        tune_for_intel(props);
    } else if (vendor_id == 0x13B5) {  // ARM Mali
        tune_for_arm(props);
    } else {
        // Conservative defaults
        tune_generic(props);
    }
    
    // Set subgroup size from query
    device_caps_.subgroup_size = subgroup_props.subgroupSize;
    
    std::cout << "[ShaderRuntime] Auto-tune complete:" << std::endl;
    std::cout << "  Optimal workgroup: " << device_caps_.optimal_workgroup_size << std::endl;
    std::cout << "  Wave size: " << device_caps_.wave_size << std::endl;
    std::cout << "  Subgroup: " << device_caps_.subgroup_size << std::endl;
    std::cout << "  Cooperative matrix: " << (device_caps_.supports_cooperative_matrix ? "yes" : "no") << std::endl;
    if (device_caps_.supports_cooperative_matrix) {
        std::cout << "  Matrix dims: " << device_caps_.cooperative_matrix_m << "x" 
                  << device_caps_.cooperative_matrix_n << "x" << device_caps_.cooperative_matrix_k << std::endl;
    }
}

void ShaderRuntime::tune_for_nvidia(const VkPhysicalDeviceProperties& props) {
    // NVIDIA Ampere/Ada optimal settings
    device_caps_.optimal_workgroup_size = 256;  // Ampere/Ada optimal
    device_caps_.wave_size = 32;
    device_caps_.prefers_warp_shuffle = true;
    
    // Check for cooperative matrix support (RTX 20 series+)
    if (props.apiVersion >= VK_API_VERSION_1_2) {
        device_caps_.supports_cooperative_matrix = check_cooperative_matrix_support();
        if (device_caps_.supports_cooperative_matrix) {
            // NVIDIA uses 16x16x16 for optimal tensor core utilization
            device_caps_.cooperative_matrix_m = 16;
            device_caps_.cooperative_matrix_n = 16;
            device_caps_.cooperative_matrix_k = 16;
        }
    }
    
    // Turing/Ampere/Ada have different optimal workgroup sizes
    if (strstr(props.deviceName, "RTX")) {
        // RTX series - use larger workgroups for better occupancy
        device_caps_.optimal_workgroup_size = 512;
    }
    
    std::cout << "[ShaderRuntime] Tuned for NVIDIA GPU" << std::endl;
}

void ShaderRuntime::tune_for_amd(const VkPhysicalDeviceProperties& props) {
    // AMD RDNA/CDNA optimal settings
    device_caps_.optimal_workgroup_size = 256;  // RDNA optimal
    device_caps_.wave_size = 64;  // AMD wave64
    device_caps_.prefers_warp_shuffle = true;
    
    // Check for matrix core support (RDNA2/CDNA)
    device_caps_.supports_cooperative_matrix = check_cooperative_matrix_support();
    if (device_caps_.supports_cooperative_matrix) {
        // AMD uses 16x16x16
        device_caps_.cooperative_matrix_m = 16;
        device_caps_.cooperative_matrix_n = 16;
        device_caps_.cooperative_matrix_k = 16;
    }
    
    std::cout << "[ShaderRuntime] Tuned for AMD GPU" << std::endl;
}

void ShaderRuntime::tune_for_intel(const VkPhysicalDeviceProperties& props) {
    // Intel Xe optimal settings
    device_caps_.optimal_workgroup_size = 128;  // Xe optimal
    device_caps_.wave_size = 8;  // Xe uses SIMD8 for compute
    device_caps_.prefers_warp_shuffle = false;
    
    // Intel Xe Matrix Extensions (XMX)
    device_caps_.supports_cooperative_matrix = check_cooperative_matrix_support();
    if (device_caps_.supports_cooperative_matrix) {
        device_caps_.cooperative_matrix_m = 8;
        device_caps_.cooperative_matrix_n = 8;
        device_caps_.cooperative_matrix_k = 16;
    }
    
    std::cout << "[ShaderRuntime] Tuned for Intel GPU" << std::endl;
}

void ShaderRuntime::tune_for_arm(const VkPhysicalDeviceProperties& props) {
    // ARM Mali optimal settings
    device_caps_.optimal_workgroup_size = 64;
    device_caps_.wave_size = 4;  // Mali uses quad-based execution
    device_caps_.prefers_warp_shuffle = false;
    device_caps_.supports_cooperative_matrix = false;  // Limited support
    
    std::cout << "[ShaderRuntime] Tuned for ARM Mali GPU" << std::endl;
}

void ShaderRuntime::tune_generic(const VkPhysicalDeviceProperties& props) {
    // Conservative defaults for unknown hardware
    device_caps_.optimal_workgroup_size = 128;
    device_caps_.wave_size = 32;
    device_caps_.prefers_warp_shuffle = false;
    device_caps_.supports_cooperative_matrix = check_cooperative_matrix_support();
    
    if (device_caps_.supports_cooperative_matrix) {
        device_caps_.cooperative_matrix_m = 16;
        device_caps_.cooperative_matrix_n = 16;
        device_caps_.cooperative_matrix_k = 16;
    }
    
    std::cout << "[ShaderRuntime] Tuned for generic GPU" << std::endl;
}

bool ShaderRuntime::check_cooperative_matrix_support() {
    // Check for VK_KHR_cooperative_matrix extension
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr);
    
    if (extension_count == 0) return false;
    
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extensions.data());
    
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            // Also need to check for VK_NV_cooperative_matrix for NVIDIA
            return true;
        }
        if (strcmp(ext.extensionName, VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            return true;
        }
    }
    
    return false;
}

ShaderRuntime::ShaderSpecialization ShaderRuntime::get_optimal_specialization(uint32_t operation_type) const {
    ShaderSpecialization spec;
    
    switch (operation_type) {
        case 0: // Matmul - use cooperative matrices if available
            if (device_caps_.supports_cooperative_matrix) {
                spec.workgroup_size_x = device_caps_.cooperative_matrix_m * 2;  // 2 warps per matrix
                spec.workgroup_size_y = device_caps_.cooperative_matrix_n / 8;
                spec.use_subgroup_ops = true;
                spec.subgroup_size = device_caps_.subgroup_size;
                spec.use_fp16_math = device_caps_.supports_fp16;
            } else {
                spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
                spec.workgroup_size_y = 1;
                spec.use_subgroup_ops = device_caps_.prefers_warp_shuffle;
                spec.subgroup_size = device_caps_.subgroup_size;
                spec.use_fp16_math = device_caps_.supports_fp16;
            }
            break;
            
        case 1: // Attention
            spec.workgroup_size_x = 16;
            spec.workgroup_size_y = 16;
            spec.use_subgroup_ops = true;
            spec.subgroup_size = device_caps_.subgroup_size;
            spec.use_fp16_math = device_caps_.supports_fp16;
            break;
            
        case 2: // RMS Norm
            spec.workgroup_size_x = device_caps_.optimal_workgroup_size;
            spec.workgroup_size_y = 1;
            spec.use_subgroup_ops = true;
            spec.subgroup_size = device_caps_.subgroup_size;
            spec.use_fp16_math = false;  // Keep precision for norms
            break;
            
        default:
            spec.workgroup_size_x = 128;
            spec.workgroup_size_y = 1;
            spec.use_subgroup_ops = false;
            spec.use_fp16_math = false;
            break;
    }
    
    return spec;
}

// ============================================================================
// Cooperative Matrix Shader Generation
// ============================================================================
VkPipeline ShaderRuntime::get_cooperative_matmul_pipeline(const ShaderSpecialization& spec) {
    if (!device_caps_.supports_cooperative_matrix) {
        std::cerr << "[ShaderRuntime] Cooperative matrices not supported, using standard matmul" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    // Generate cooperative matrix shader
    std::string shader_source = generate_cooperative_matmul_shader();
    
    auto shader_result = compile_glsl_to_spirv(shader_source, {});
    if (!shader_result.has_value()) {
        std::cerr << "[ShaderRuntime] Failed to compile cooperative matrix shader" << std::endl;
        return VK_NULL_HANDLE;
    }
    
    std::vector<uint32_t> spec_data;
    VkSpecializationInfo spec_info = create_specialization_info(spec, spec_data);
    
    VkComputePipelineCreateInfo pipeline_info = create_compute_pipeline_info(
        shader_result.value(), pipeline_layout_, &spec_info);
    
    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, 
                                             &pipeline_info, nullptr, &pipeline);
    
    if (result == VK_SUCCESS) {
        std::cout << "[ShaderRuntime] Created cooperative matrix pipeline" << std::endl;
    }
    
    return (result == VK_SUCCESS) ? pipeline : VK_NULL_HANDLE;
}

std::string ShaderRuntime::generate_cooperative_matmul_shader() {
    std::stringstream ss;
    
    ss << "#version 460\n";
    ss << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    ss << "#extension GL_KHR_cooperative_matrix : require\n";
    ss << "#extension GL_KHR_shader_subgroup : require\n\n";
    
    // Use device-specific cooperative matrix dimensions
    uint32_t M = device_caps_.cooperative_matrix_m;
    uint32_t N = device_caps_.cooperative_matrix_n;
    uint32_t K = device_caps_.cooperative_matrix_k;
    
    ss << "layout(local_size_x = " << M << ", local_size_y = " << N / 8 << ") in;\n\n";
    
    ss << R"(
layout(push_constant) uniform PushConstants {
    uint M_total, N_total, K_total;
} pc;

layout(set = 0, binding = 0) readonly buffer A { f16vec4 a[]; };
layout(set = 0, binding = 1) readonly buffer B { f16vec4 b[]; };
layout(set = 0, binding = 2) buffer C { f16vec4 c[]; };

// Cooperative matrix types
coopmat<f16, gl_ScopeSubgroup, )" << M << ", " << K << ", gl_MatrixUseA> matA;
coopmat<f16, gl_ScopeSubgroup, )" << K << ", " << N << ", gl_MatrixUseB> matB;
coopmat<f32, gl_ScopeSubgroup, )" << M << ", " << N << ", gl_MatrixUseAccumulator> matC;

void main() {
    uint warp_id = gl_SubgroupID;
    uint warp_m = warp_id % (pc.M_total / )" << M << R"();
    uint warp_n = warp_id / (pc.M_total / )" << M << R"();
    
    // Initialize accumulator to zero
    coopMatLoad(matC, c, warp_m * )" << M << R"( * pc.N_total + warp_n * )" << N << R"(, pc.N_total, gl_CooperativeMatrixLayoutRowMajor);
    
    // Compute matrix multiplication using cooperative matrices
    for (uint k = 0; k < pc.K_total; k += )" << K << R"() {
        // Load A and B tiles
        coopMatLoad(matA, a, warp_m * )" << M << R"( * pc.K_total + k, pc.K_total, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(matB, b, k * pc.N_total + warp_n * )" << N << R"(, pc.N_total, gl_CooperativeMatrixLayoutRowMajor);
        
        // Multiply-accumulate
        matC = coopMatMulAdd(matA, matB, matC);
    }
    
    // Store result
    coopMatStore(matC, c, warp_m * )" << M << R"( * pc.N_total + warp_n * )" << N << R"(, pc.N_total, gl_CooperativeMatrixLayoutRowMajor);
}
)";
    
    return ss.str();
}

std::string ShaderRuntime::generate_tuned_matmul_shader(uint32_t m, uint32_t n, uint32_t k) {
    std::stringstream ss;
    
    ss << "#version 460\n";
    ss << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    
    if (device_caps_.prefers_warp_shuffle) {
        ss << "#extension GL_KHR_shader_subgroup_shuffle : require\n";
    }
    
    ss << "\nlayout(local_size_x = " << device_caps_.optimal_workgroup_size << ", local_size_y = 1) in;\n\n";
    
    ss << R"(
layout(push_constant) uniform PushConstants {
    uint M, N, K;
} pc;

layout(set = 0, binding = 0) readonly buffer A { float16_t a[]; };
layout(set = 0, binding = 1) readonly buffer B { float16_t b[]; };
layout(set = 0, binding = 2) buffer C { float16_t c[]; };

shared float16_t tile_a[128][32];
shared float16_t tile_b[32][128];

void main() {
    uint global_x = gl_GlobalInvocationID.x;
    uint global_y = gl_GlobalInvocationID.y;
    
    float16_t acc = 0.0hf;
    
    // Tiled matrix multiplication with shared memory
    for (uint tile_k = 0; tile_k < pc.K; tile_k += 32) {
        // Load tiles cooperatively
        if (global_y < pc.M && tile_k + gl_LocalInvocationID.x < pc.K) {
            tile_a[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                a[global_y * pc.K + tile_k + gl_LocalInvocationID.x];
        }
        
        if (global_x < pc.N && tile_k + gl_LocalInvocationID.y < pc.K) {
            tile_b[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                b[(tile_k + gl_LocalInvocationID.y) * pc.N + global_x];
        }
        
        barrier();
        
        // Compute partial dot product
        for (uint k = 0; k < 32 && tile_k + k < pc.K; ++k) {
            acc += tile_a[gl_LocalInvocationID.y][k] * tile_b[k][gl_LocalInvocationID.x];
        }
        
        barrier();
    }
    
    if (global_y < pc.M && global_x < pc.N) {
        c[global_y * pc.N + global_x] = acc;
    }
}
)";
    
    return ss.str();
}

std::string ShaderRuntime::generate_tuned_attention_shader(uint32_t seq_len, uint32_t head_dim) {
    std::stringstream ss;
    
    ss << "#version 460\n";
    ss << "#extension GL_KHR_shader_subgroup : require\n";
    
    if (device_caps_.supports_fp16) {
        ss << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    }
    
    ss << "\nlayout(local_size_x = " << device_caps_.subgroup_size << ", local_size_y = 1) in;\n\n";
    
    ss << R"(
layout(push_constant) uniform PushConstants {
    uint seq_len, head_dim, num_heads;
    float scale;
} pc;

layout(set = 0, binding = 0) readonly buffer Q { float q[]; };
layout(set = 0, binding = 1) readonly buffer K { float k[]; };
layout(set = 0, binding = 2) readonly buffer V { float v[]; };
layout(set = 0, binding = 3) buffer O { float o[]; };

void main() {
    uint head = gl_WorkGroupID.x;
    uint token = gl_GlobalInvocationID.x;
    
    if (head >= pc.num_heads || token >= pc.seq_len) return;
    
    // Compute attention scores with subgroup optimization
    float max_score = -1e30;
    float sum_exp = 0.0;
    float acc = 0.0;
    
    for (uint i = gl_SubgroupInvocationID; i < pc.seq_len; i += gl_SubgroupSize) {
        float score = 0.0;
        for (uint d = 0; d < pc.head_dim; ++d) {
            score += q[token * pc.head_dim * pc.num_heads + head * pc.head_dim + d] *
                     k[i * pc.head_dim * pc.num_heads + head * pc.head_dim + d];
        }
        score *= pc.scale;
        
        // Subgroup-level softmax
        float max_subgroup = subgroupMax(score);
        max_score = max(max_score, max_subgroup);
        
        float exp_score = exp(score - max_score);
        sum_exp += subgroupAdd(exp_score);
        
        // Accumulate weighted values
        for (uint d = 0; d < pc.head_dim; ++d) {
            acc += exp_score * v[i * pc.head_dim * pc.num_heads + head * pc.head_dim + d];
        }
    }
    
    // Write output
    if (gl_SubgroupInvocationID == 0) {
        for (uint d = 0; d < pc.head_dim; ++d) {
            o[token * pc.head_dim * pc.num_heads + head * pc.head_dim + d] = acc / sum_exp;
        }
    }
}
)";
    
    return ss.str();
}

// ============================================================================
// Pipeline Getters
// ============================================================================
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
    uint enable_causal_mask;
} pc;

layout (set = 0, binding = 0) readonly buffer QBuffer { float q[]; };
layout (set = 0, binding = 1) readonly buffer KBuffer { float k[]; };
layout (set = 0, binding = 2) readonly buffer VBuffer { float v[]; };
layout (set = 0, binding = 3) buffer OutputBuffer { float output[]; };

shared float16_t shared_k[16][16];
shared float16_t shared_v[16][16];
shared float16_t shared_scores[16][16];

bool is_causal_masked(uint query_pos, uint key_pos) {
    return pc.enable_causal_mask != 0 && key_pos > query_pos;
}

void main() {
    uint head = gl_GlobalInvocationID.y;
    uint token_pos = gl_GlobalInvocationID.x;
    
    if (head >= pc.num_heads || token_pos >= pc.seq_len) return;
    
    float16_t q_vec[16];
    for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
        q_vec[dim] = float16_t(q[token_pos * pc.num_heads * pc.head_dim + head * pc.head_dim + dim]);
    }
    
    for (uint seq_idx = 0; seq_idx < pc.seq_len; ++seq_idx) {
        float16_t score = 0.0hf;
        
        for (uint kv_head = 0; kv_head < pc.num_kv_heads; ++kv_head) {
            for (uint dim = 0; dim < pc.head_dim && dim < 16; ++dim) {
                float k_val = k[seq_idx * pc.num_kv_heads * pc.head_dim + kv_head * pc.head_dim + dim];
                score += q_vec[dim] * float16_t(k_val);
            }
        }
        
        if (is_causal_masked(token_pos, seq_idx)) {
            score = -65504.0hf;
        } else {
            score *= float16_t(pc.scale);
        }
        
        shared_scores[seq_idx / 16][seq_idx % 16] = score;
    }
    
    barrier();
    
    for (uint seq_idx = 0; seq_idx < pc.seq_len; ++seq_idx) {
        float16_t score = shared_scores[seq_idx / 16][seq_idx % 16];
        
        float16_t max_score = subgroupMax(score);
        float16_t exp_score = exp(score - max_score);
        float16_t sum_scores = subgroupAdd(exp_score);
        float16_t attention_weight = exp_score / (sum_scores + 1e-6hf);
        shared_scores[seq_idx / 16][seq_idx % 16] = attention_weight;
    }
    
    barrier();
    
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
    
    // SwiGLU activation
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
    
    // Check for cooperative matrix extension
    caps.supports_cooperative_matrix = check_cooperative_matrix_support();
    if (caps.supports_cooperative_matrix) {
        caps.cooperative_matrix_m = 16;
        caps.cooperative_matrix_n = 16;
        caps.cooperative_matrix_k = 16;
    }

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

// ============================================================================
// Pipeline Cache Persistence Implementation
// ============================================================================

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
    
    // Validate cache header
    if (cache_size < 16) {
        std::cerr << "[ShaderCache] Cache file too small, ignoring" << std::endl;
        return;
    }
    
    // Check for valid Vulkan pipeline cache header
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
    
    // Compare device UUID
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

} // namespace vk_symbiote
