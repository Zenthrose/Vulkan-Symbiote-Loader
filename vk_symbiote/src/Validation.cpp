// ============================================================================
// PRIORITY 3: Validation Implementation
// ============================================================================

#include "Validation.h"
#include <iostream>
#include <cassert>
#include <cstring>

namespace vk_symbiote {

// ============================================================================
// Static Member Definitions
// ============================================================================

std::unordered_map<VkDeviceMemory, Validation::AllocationInfo> Validation::allocation_map_;
std::unordered_map<VkBuffer, Validation::BufferInfo> Validation::buffer_map_;
std::mutex Validation::allocation_mutex_;
std::mutex Validation::buffer_mutex_;
Validation::MemoryStats Validation::memory_stats_;

std::unordered_map<std::string, UnitTest::TestFunc> UnitTest::tests_;
std::mutex UnitTest::test_mutex_;

// ============================================================================
// VkResult to String Conversion
// ============================================================================

const char* Validation::result_to_string(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        default: return "UNKNOWN_VK_RESULT";
    }
}

// ============================================================================
// Memory Statistics
// ============================================================================

void Validation::MemoryStats::print() const {
    std::cout << "=== GPU Memory Statistics ===" << std::endl;
    std::cout << "Total Allocated: " << (total_allocated / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Total Used: " << (total_used / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Peak Allocated: " << (peak_allocated / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Host-Visible: " << (host_visible_allocated / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Device-Local: " << (device_local_allocated / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Allocations: " << allocation_count << std::endl;
    std::cout << "Buffers: " << buffer_count << std::endl;
    std::cout << "Images: " << image_count << std::endl;
}

// ============================================================================
// Memory Tracking
// ============================================================================

void Validation::track_allocation(VkDeviceMemory memory, VkDeviceSize size,
                                   uint32_t memory_type, VkMemoryPropertyFlags properties) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    AllocationInfo info;
    info.memory = memory;
    info.size = size;
    info.memory_type = memory_type;
    info.properties = properties;
    info.timestamp = std::chrono::steady_clock::now();
    
    allocation_map_[memory] = info;
    
    // Update statistics
    memory_stats_.total_allocated += size;
    memory_stats_.total_used += size;
    memory_stats_.allocation_count++;
    
    if (memory_stats_.total_allocated > memory_stats_.peak_allocated) {
        memory_stats_.peak_allocated = memory_stats_.total_allocated;
    }
    
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        memory_stats_.host_visible_allocated += size;
    }
    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        memory_stats_.device_local_allocated += size;
    }
}

void Validation::track_deallocation(VkDeviceMemory memory) {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    
    auto it = allocation_map_.find(memory);
    if (it != allocation_map_.end()) {
        VkDeviceSize size = it->second.size;
        VkMemoryPropertyFlags properties = it->second.properties;
        
        memory_stats_.total_allocated -= size;
        memory_stats_.allocation_count--;
        
        if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            memory_stats_.host_visible_allocated -= size;
        }
        if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            memory_stats_.device_local_allocated -= size;
        }
        
        allocation_map_.erase(it);
    }
}

void Validation::track_buffer_creation(VkBuffer buffer, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    BufferInfo info;
    info.buffer = buffer;
    info.size = size;
    info.creation_time = std::chrono::steady_clock::now();
    
    buffer_map_[buffer] = info;
    memory_stats_.buffer_count++;
}

void Validation::track_buffer_destruction(VkBuffer buffer) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    if (buffer_map_.erase(buffer) > 0) {
        memory_stats_.buffer_count--;
    }
}

Validation::MemoryStats Validation::get_memory_stats() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    return memory_stats_;
}

void Validation::reset_memory_stats() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    memory_stats_ = MemoryStats{};
}

bool Validation::check_memory_budget(VkPhysicalDevice physical_device, VkDeviceSize requested_size) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    
    // Check if we can allocate from any heap
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        uint32_t heap_index = mem_props.memoryTypes[i].heapIndex;
        VkDeviceSize heap_size = mem_props.memoryHeaps[heap_index].size;
        
        // Leave 10% margin for safety
        if (heap_size * 0.9 > requested_size) {
            return true;
        }
    }
    
    return false;
}

// ============================================================================
// Buffer Validation
// ============================================================================

void Validation::register_buffer(const BufferInfo& info) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_map_[info.buffer] = info;
}

void Validation::unregister_buffer(VkBuffer buffer) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_map_.erase(buffer);
}

bool Validation::is_buffer_registered(VkBuffer buffer) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_map_.find(buffer) != buffer_map_.end();
}

Validation::BufferInfo Validation::get_buffer_info(VkBuffer buffer) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    auto it = buffer_map_.find(buffer);
    if (it != buffer_map_.end()) {
        return it->second;
    }
    return BufferInfo{};
}

bool Validation::validate_buffer_size(VkBuffer buffer, VkDeviceSize required_size) {
    BufferInfo info = get_buffer_info(buffer);
    return info.buffer != VK_NULL_HANDLE && info.size >= required_size;
}

bool Validation::validate_buffer_usage(VkBuffer buffer, VkBufferUsageFlags required_usage) {
    BufferInfo info = get_buffer_info(buffer);
    return info.buffer != VK_NULL_HANDLE && (info.usage & required_usage) == required_usage;
}

bool Validation::validate_buffer_memory(VkBuffer buffer) {
    BufferInfo info = get_buffer_info(buffer);
    return info.buffer != VK_NULL_HANDLE && info.memory != VK_NULL_HANDLE;
}

// ============================================================================
// Device Validation
// ============================================================================

void Validation::DeviceValidationResult::print() const {
    std::cout << "=== Device Validation Result ===" << std::endl;
    std::cout << "Suitable: " << (is_suitable ? "YES" : "NO") << std::endl;
    std::cout << "Compute Queue: " << (has_compute_queue ? "YES" : "NO") << std::endl;
    std::cout << "Storage Buffers: " << (has_storage_buffers ? "YES" : "NO") << std::endl;
    std::cout << "Push Constants: " << (has_push_constants ? "YES" : "NO") << std::endl;
    std::cout << "Subgroup Ops: " << (has_subgroup_ops ? "YES" : "NO") << std::endl;
    std::cout << "Cooperative Matrix: " << (has_cooperative_matrix ? "YES" : "NO") << std::endl;
    std::cout << "Min Memory: " << (min_memory_size / 1024 / 1024) << " MB" << std::endl;
    
    if (!missing_features.empty()) {
        std::cout << "Missing Features:" << std::endl;
        for (const auto& feature : missing_features) {
            std::cout << "  - " << feature << std::endl;
        }
    }
}

Validation::DeviceValidationResult Validation::validate_device(VkPhysicalDevice device) {
    DeviceValidationResult result;
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);
    
    // Check compute queue
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
    
    for (const auto& family : queue_families) {
        if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            result.has_compute_queue = true;
            break;
        }
    }
    
    if (!result.has_compute_queue) {
        result.missing_features.push_back("Compute queue family");
    }
    
    // Check storage buffers
    result.has_storage_buffers = true; // Required by Vulkan spec
    
    // Check push constants
    result.has_push_constants = props.limits.maxPushConstantsSize >= 128;
    if (!result.has_push_constants) {
        result.missing_features.push_back("Adequate push constant size");
    }
    
    // Check memory
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);
    
    VkDeviceSize max_heap_size = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].size > max_heap_size) {
            max_heap_size = mem_props.memoryHeaps[i].size;
        }
    }
    result.min_memory_size = max_heap_size;
    
    // Minimum 2GB for 70B models
    if (max_heap_size < 2ULL * 1024 * 1024 * 1024) {
        result.missing_features.push_back("Minimum 2GB device memory");
    }
    
    // Determine suitability
    result.is_suitable = result.has_compute_queue && 
                         result.has_push_constants &&
                         max_heap_size >= 2ULL * 1024 * 1024 * 1024;
    
    return result;
}

bool Validation::check_required_extensions(VkPhysicalDevice device,
                                            const std::vector<const char*>& extensions) {
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
    
    if (extension_count == 0) return extensions.empty();
    
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());
    
    for (const char* required : extensions) {
        bool found = false;
        for (const auto& available : available_extensions) {
            if (strcmp(required, available.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    
    return true;
}

// ============================================================================
// Pipeline Validation
// ============================================================================

bool Validation::validate_workgroup_size(uint32_t x, uint32_t y, uint32_t z,
                                         VkPhysicalDeviceLimits limits) {
    uint32_t total_invocations = x * y * z;
    
    if (total_invocations > limits.maxComputeWorkGroupInvocations) {
        std::cerr << "[Validation] Workgroup size " << x << "x" << y << "x" << z
                  << " (" << total_invocations << " total) exceeds limit ("
                  << limits.maxComputeWorkGroupInvocations << ")" << std::endl;
        return false;
    }
    
    if (x > limits.maxComputeWorkGroupSize[0] ||
        y > limits.maxComputeWorkGroupSize[1] ||
        z > limits.maxComputeWorkGroupSize[2]) {
        std::cerr << "[Validation] Workgroup dimension exceeds per-axis limit" << std::endl;
        return false;
    }
    
    return true;
}

// ============================================================================
// Debug Utilities
// ============================================================================

void Validation::dump_device_info(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    
    std::cout << "=== Device Information ===" << std::endl;
    std::cout << "Name: " << props.deviceName << std::endl;
    std::cout << "Type: " << [](VkPhysicalDeviceType t) {
        switch (t) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete GPU";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual GPU";
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
            default: return "Other";
        }
    }(props.deviceType) << std::endl;
    std::cout << "API Version: " << VK_VERSION_MAJOR(props.apiVersion) << "."
              << VK_VERSION_MINOR(props.apiVersion) << "."
              << VK_VERSION_PATCH(props.apiVersion) << std::endl;
    std::cout << "Driver Version: " << props.driverVersion << std::endl;
    std::cout << "Vendor ID: 0x" << std::hex << props.vendorID << std::dec << std::endl;
}

void Validation::dump_memory_info(VkPhysicalDevice device) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);
    
    std::cout << "=== Memory Information ===" << std::endl;
    std::cout << "Memory Types: " << mem_props.memoryTypeCount << std::endl;
    std::cout << "Memory Heaps: " << mem_props.memoryHeapCount << std::endl;
    
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        std::cout << "  Heap " << i << ": " 
                  << (mem_props.memoryHeaps[i].size / 1024 / 1024) << " MB";
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            std::cout << " (Device Local)";
        }
        std::cout << std::endl;
    }
}

void Validation::dump_queue_info(VkPhysicalDevice device) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
    
    std::cout << "=== Queue Information ===" << std::endl;
    std::cout << "Queue Families: " << queue_family_count << std::endl;
    
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        std::cout << "  Family " << i << ": ";
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) std::cout << "Graphics ";
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) std::cout << "Compute ";
        if (queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) std::cout << "Transfer ";
        std::cout << "(" << queue_families[i].queueCount << " queues)" << std::endl;
    }
}

// ============================================================================
// Unit Test Implementation
// ============================================================================

void UnitTest::TestResult::print() const {
    std::cout << (passed ? "[PASS]" : "[FAIL]") << " " << name;
    std::cout << " (" << std::fixed << std::setprecision(2) << duration_ms << " ms)";
    if (!passed && !error_message.empty()) {
        std::cout << " - " << error_message;
    }
    std::cout << std::endl;
}

void UnitTest::register_test(const std::string& name, TestFunc func) {
    std::lock_guard<std::mutex> lock(test_mutex_);
    tests_[name] = func;
}

std::vector<UnitTest::TestResult> UnitTest::run_all_tests() {
    std::vector<TestResult> results;
    
    std::lock_guard<std::mutex> lock(test_mutex_);
    for (const auto& [name, func] : tests_) {
        auto start = std::chrono::steady_clock::now();
        TestResult result = func();
        auto end = std::chrono::steady_clock::now();
        
        result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        results.push_back(result);
    }
    
    return results;
}

std::vector<UnitTest::TestResult> UnitTest::run_tests_by_prefix(const std::string& prefix) {
    std::vector<TestResult> results;
    
    std::lock_guard<std::mutex> lock(test_mutex_);
    for (const auto& [name, func] : tests_) {
        if (name.substr(0, prefix.length()) == prefix) {
            auto start = std::chrono::steady_clock::now();
            TestResult result = func();
            auto end = std::chrono::steady_clock::now();
            
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            results.push_back(result);
        }
    }
    
    return results;
}

// Assertions throw exceptions to stop test execution
class TestAssertionFailed : public std::exception {
public:
    TestAssertionFailed(const std::string& msg) : message_(msg) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

void UnitTest::assert_true(bool condition, const std::string& message) {
    if (!condition) {
        throw TestAssertionFailed("Expected true: " + message);
    }
}

void UnitTest::assert_false(bool condition, const std::string& message) {
    if (condition) {
        throw TestAssertionFailed("Expected false: " + message);
    }
}

void UnitTest::assert_not_null(void* ptr, const std::string& message) {
    if (ptr == nullptr) {
        throw TestAssertionFailed("Expected non-null: " + message);
    }
}

void UnitTest::assert_vk_success(VkResult result, const std::string& message) {
    if (result != VK_SUCCESS) {
        throw TestAssertionFailed("VK call failed (" + 
            std::string(Validation::result_to_string(result)) + "): " + message);
    }
}

} // namespace vk_symbiote
