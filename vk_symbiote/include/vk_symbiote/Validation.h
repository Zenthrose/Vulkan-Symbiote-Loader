#pragma once

// ============================================================================
// PRIORITY 3: Validation and Error Checking System
// ============================================================================
// Provides comprehensive VkResult checking, GPU memory validation, and
// debugging utilities for the Vulkan Symbiote engine.

#include "Common.h"
#include <vulkan/vulkan.h>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace vk_symbiote {

// ============================================================================
// VkResult Error Checking Macros
// ============================================================================

// Basic check - returns error code on failure
#define VK_CHECK(result, msg) \
    do { \
        VkResult vk_result = (result); \
        if (vk_result != VK_SUCCESS) { \
            std::cerr << "[VK_CHECK] " << msg << " failed: " \
                      << vk_symbiote::Validation::result_to_string(vk_result) \
                      << " (" << vk_result << ")" << std::endl; \
            return vk_result; \
        } \
    } while(0)

// Check with custom return value
#define VK_CHECK_RETURN(result, msg, ret) \
    do { \
        VkResult vk_result = (result); \
        if (vk_result != VK_SUCCESS) { \
            std::cerr << "[VK_CHECK] " << msg << " failed: " \
                      << vk_symbiote::Validation::result_to_string(vk_result) \
                      << " (" << vk_result << ")" << std::endl; \
            return ret; \
        } \
    } while(0)

// Check that logs but doesn't return (for cleanup paths)
#define VK_CHECK_LOG(result, msg) \
    do { \
        VkResult vk_result = (result); \
        if (vk_result != VK_SUCCESS) { \
            std::cerr << "[VK_CHECK] " << msg << " failed: " \
                      << vk_symbiote::Validation::result_to_string(vk_result) \
                      << " (" << vk_result << ")" << std::endl; \
        } \
    } while(0)

// Assert check - aborts on failure in debug builds
#ifdef NDEBUG
#define VK_ASSERT(result, msg) (void)(result)
#else
#define VK_ASSERT(result, msg) \
    do { \
        VkResult vk_result = (result); \
        if (vk_result != VK_SUCCESS) { \
            std::cerr << "[VK_ASSERT] " << msg << " failed: " \
                      << vk_symbiote::Validation::result_to_string(vk_result) \
                      << " (" << vk_result << ")" << std::endl; \
            assert(false && msg); \
        } \
    } while(0)
#endif

// ============================================================================
// Validation Utilities Class
// ============================================================================

class Validation {
public:
    // Convert VkResult to human-readable string
    static const char* result_to_string(VkResult result);
    
    // Check if VkResult indicates success
    static bool is_success(VkResult result) { return result == VK_SUCCESS; }
    static bool is_error(VkResult result) { return result < VK_SUCCESS; }
    static bool is_warning(VkResult result) { return result > VK_SUCCESS; }
    
    // ============================================================================
    // GPU Memory Validation
    // ============================================================================
    
    struct MemoryStats {
        VkDeviceSize total_allocated = 0;
        VkDeviceSize total_used = 0;
        VkDeviceSize peak_allocated = 0;
        VkDeviceSize host_visible_allocated = 0;
        VkDeviceSize device_local_allocated = 0;
        uint32_t allocation_count = 0;
        uint32_t buffer_count = 0;
        uint32_t image_count = 0;
        
        void print() const;
    };
    
    // Track GPU memory allocations
    static void track_allocation(VkDeviceMemory memory, VkDeviceSize size, 
                                  uint32_t memory_type, VkMemoryPropertyFlags properties);
    static void track_deallocation(VkDeviceMemory memory);
    static void track_buffer_creation(VkBuffer buffer, VkDeviceSize size);
    static void track_buffer_destruction(VkBuffer buffer);
    
    // Get current memory statistics
    static MemoryStats get_memory_stats();
    static void reset_memory_stats();
    
    // Validate memory budget against device limits
    static bool check_memory_budget(VkPhysicalDevice physical_device, VkDeviceSize requested_size);
    
    // ============================================================================
    // Buffer Validation
    // ============================================================================
    
    struct BufferInfo {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkBufferUsageFlags usage = 0;
        VkMemoryPropertyFlags properties = 0;
        bool is_mapped = false;
        void* mapped_ptr = nullptr;
        std::chrono::steady_clock::time_point creation_time;
        const char* debug_name = nullptr;
    };
    
    // Register buffer for tracking
    static void register_buffer(const BufferInfo& info);
    static void unregister_buffer(VkBuffer buffer);
    static bool is_buffer_registered(VkBuffer buffer);
    static BufferInfo get_buffer_info(VkBuffer buffer);
    
    // Validate buffer operations
    static bool validate_buffer_size(VkBuffer buffer, VkDeviceSize required_size);
    static bool validate_buffer_usage(VkBuffer buffer, VkBufferUsageFlags required_usage);
    static bool validate_buffer_memory(VkBuffer buffer);
    
    // ============================================================================
    // Shader Validation
    // ============================================================================
    
    struct ShaderValidationResult {
        bool is_valid = false;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        uint32_t workgroup_size_x = 0;
        uint32_t workgroup_size_y = 0;
        uint32_t workgroup_size_z = 0;
        uint32_t required_shared_memory = 0;
        
        void print() const;
    };
    
    // Validate SPIR-V shader before creating pipeline
    static ShaderValidationResult validate_shader(const std::vector<uint32_t>& spirv_code,
                                                   VkPhysicalDeviceLimits limits);
    static ShaderValidationResult validate_shader_glsl(const std::string& glsl_source,
                                                        VkPhysicalDeviceLimits limits);
    
    // ============================================================================
    // Descriptor Set Validation
    // ============================================================================
    
    struct DescriptorSetValidation {
        bool is_valid = false;
        std::vector<std::string> errors;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        
        void print() const;
    };
    
    static DescriptorSetValidation validate_descriptor_set_layout(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings,
        VkPhysicalDeviceLimits limits);
    
    // ============================================================================
    // Pipeline Validation
    // ============================================================================
    
    static bool validate_compute_pipeline(VkPipeline pipeline, VkPipelineLayout layout);
    static bool validate_workgroup_size(uint32_t x, uint32_t y, uint32_t z,
                                        VkPhysicalDeviceLimits limits);
    
    // ============================================================================
    // Device Capability Validation
    // ============================================================================
    
    struct DeviceValidationResult {
        bool is_suitable = false;
        bool has_compute_queue = false;
        bool has_storage_buffers = false;
        bool has_push_constants = false;
        bool has_subgroup_ops = false;
        bool has_cooperative_matrix = false;
        VkDeviceSize min_memory_size = 0;
        std::vector<std::string> missing_features;
        
        void print() const;
    };
    
    static DeviceValidationResult validate_device(VkPhysicalDevice device);
    static bool check_required_extensions(VkPhysicalDevice device, 
                                          const std::vector<const char*>& extensions);
    
    // ============================================================================
    // Debug Utilities
    // ============================================================================
    
    // Set debug name for Vulkan objects (if VK_EXT_debug_utils available)
    static void set_debug_name(VkDevice device, VkObjectType type, uint64_t handle,
                                const char* name);
    static void begin_debug_label(VkCommandBuffer cmd, const char* label);
    static void end_debug_label(VkCommandBuffer cmd);
    
    // Dump system information
    static void dump_device_info(VkPhysicalDevice device);
    static void dump_memory_info(VkPhysicalDevice device);
    static void dump_queue_info(VkPhysicalDevice device);
    
private:
    // Internal tracking structures
    struct AllocationInfo {
        VkDeviceMemory memory;
        VkDeviceSize size;
        uint32_t memory_type;
        VkMemoryPropertyFlags properties;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    static std::unordered_map<VkDeviceMemory, AllocationInfo> allocation_map_;
    static std::unordered_map<VkBuffer, BufferInfo> buffer_map_;
    static std::mutex allocation_mutex_;
    static std::mutex buffer_mutex_;
    static MemoryStats memory_stats_;
};

// ============================================================================
// Scoped Validation Helpers
// ============================================================================

class ScopedBufferValidation {
public:
    ScopedBufferValidation(VkBuffer buffer, const char* operation);
    ~ScopedBufferValidation();
    
private:
    VkBuffer buffer_;
    const char* operation_;
    Validation::BufferInfo info_;
};

class ScopedGpuTimer {
public:
    ScopedGpuTimer(VkDevice device, VkCommandBuffer cmd, const char* name);
    ~ScopedGpuTimer();
    
    double get_elapsed_ms() const;
    
private:
    VkDevice device_;
    VkCommandBuffer cmd_;
    const char* name_;
    VkQueryPool query_pool_;
    uint32_t query_index_;
    std::chrono::steady_clock::time_point cpu_start_;
};

// ============================================================================
// Unit Test Framework
// ============================================================================

class UnitTest {
public:
    struct TestResult {
        bool passed = false;
        std::string name;
        std::string error_message;
        double duration_ms = 0.0;
        
        void print() const;
    };
    
    using TestFunc = std::function<TestResult()>;
    
    // Register and run tests
    static void register_test(const std::string& name, TestFunc func);
    static std::vector<TestResult> run_all_tests();
    static std::vector<TestResult> run_tests_by_prefix(const std::string& prefix);
    
    // Assertions
    static void assert_true(bool condition, const std::string& message);
    static void assert_false(bool condition, const std::string& message);
    static void assert_equals(auto expected, auto actual, const std::string& message);
    static void assert_not_null(void* ptr, const std::string& message);
    static void assert_vk_success(VkResult result, const std::string& message);
    
private:
    static std::unordered_map<std::string, TestFunc> tests_;
    static std::mutex test_mutex_;
};

// ============================================================================
// Test Macros
// ============================================================================

#define TEST(name) \
    static void test_##name(); \
    static struct test_registrar_##name { \
        test_registrar_##name() { \
            vk_symbiote::UnitTest::register_test(#name, test_##name); \
        } \
    } test_instance_##name; \
    static void test_##name()

#define ASSERT_TRUE(cond) vk_symbiote::UnitTest::assert_true(cond, #cond)
#define ASSERT_FALSE(cond) vk_symbiote::UnitTest::assert_false(cond, #cond)
#define ASSERT_EQ(expected, actual) vk_symbiote::UnitTest::assert_equals(expected, actual, #expected " == " #actual)
#define ASSERT_NOT_NULL(ptr) vk_symbiote::UnitTest::assert_not_null(ptr, #ptr " != nullptr")
#define ASSERT_VK_SUCCESS(result) vk_symbiote::UnitTest::assert_vk_success(result, #result)

} // namespace vk_symbiote
