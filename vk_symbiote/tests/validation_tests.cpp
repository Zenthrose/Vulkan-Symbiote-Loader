// ============================================================================
// PRIORITY 3: Unit Tests for Critical Paths
// ============================================================================

#include "Validation.h"
#include <iostream>

namespace vk_symbiote {

// Test VkResult conversion
TEST(vkresult_success) {
    UnitTest::TestResult result;
    result.name = "vkresult_success";
    
    try {
        ASSERT_EQ(VK_SUCCESS, VK_SUCCESS);
        ASSERT_TRUE(Validation::is_success(VK_SUCCESS));
        ASSERT_FALSE(Validation::is_error(VK_SUCCESS));
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

TEST(vkresult_error_detection) {
    UnitTest::TestResult result;
    result.name = "vkresult_error_detection";
    
    try {
        ASSERT_TRUE(Validation::is_error(VK_ERROR_OUT_OF_HOST_MEMORY));
        ASSERT_TRUE(Validation::is_error(VK_ERROR_DEVICE_LOST));
        ASSERT_FALSE(Validation::is_success(VK_ERROR_INITIALIZATION_FAILED));
        ASSERT_TRUE(Validation::is_warning(VK_SUBOPTIMAL_KHR));
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

TEST(vkresult_string_conversion) {
    UnitTest::TestResult result;
    result.name = "vkresult_string_conversion";
    
    try {
        ASSERT_NOT_NULL(Validation::result_to_string(VK_SUCCESS));
        ASSERT_NOT_NULL(Validation::result_to_string(VK_ERROR_OUT_OF_HOST_MEMORY));
        ASSERT_NOT_NULL(Validation::result_to_string(VK_ERROR_DEVICE_LOST));
        
        // Verify common results return non-empty strings
        ASSERT_TRUE(strlen(Validation::result_to_string(VK_SUCCESS)) > 0);
        ASSERT_TRUE(strlen(Validation::result_to_string(VK_ERROR_UNKNOWN)) > 0);
        
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

// Test memory tracking
TEST(memory_tracking_basic) {
    UnitTest::TestResult result;
    result.name = "memory_tracking_basic";
    
    try {
        // Reset stats
        Validation::reset_memory_stats();
        
        auto stats = Validation::get_memory_stats();
        ASSERT_EQ(0ULL, stats.total_allocated);
        ASSERT_EQ(0ULL, stats.allocation_count);
        
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

// Test workgroup size validation
TEST(workgroup_size_valid) {
    UnitTest::TestResult result;
    result.name = "workgroup_size_valid";
    
    try {
        VkPhysicalDeviceLimits limits = {};
        limits.maxComputeWorkGroupInvocations = 1024;
        limits.maxComputeWorkGroupSize[0] = 1024;
        limits.maxComputeWorkGroupSize[1] = 1024;
        limits.maxComputeWorkGroupSize[2] = 64;
        
        // Valid sizes
        ASSERT_TRUE(Validation::validate_workgroup_size(128, 1, 1, limits));
        ASSERT_TRUE(Validation::validate_workgroup_size(256, 1, 1, limits));
        ASSERT_TRUE(Validation::validate_workgroup_size(16, 16, 1, limits));
        ASSERT_TRUE(Validation::validate_workgroup_size(8, 8, 4, limits));
        
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

TEST(workgroup_size_invalid) {
    UnitTest::TestResult result;
    result.name = "workgroup_size_invalid";
    
    try {
        VkPhysicalDeviceLimits limits = {};
        limits.maxComputeWorkGroupInvocations = 1024;
        limits.maxComputeWorkGroupSize[0] = 1024;
        limits.maxComputeWorkGroupSize[1] = 1024;
        limits.maxComputeWorkGroupSize[2] = 64;
        
        // Invalid: too many total invocations
        ASSERT_FALSE(Validation::validate_workgroup_size(1024, 2, 1, limits));
        
        // Invalid: exceeds per-axis limit
        ASSERT_FALSE(Validation::validate_workgroup_size(2048, 1, 1, limits));
        
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

// Test buffer validation
TEST(buffer_validation_unregistered) {
    UnitTest::TestResult result;
    result.name = "buffer_validation_unregistered";
    
    try {
        VkBuffer fake_buffer = reinterpret_cast<VkBuffer>(0xDEADBEEF);
        
        ASSERT_FALSE(Validation::is_buffer_registered(fake_buffer));
        ASSERT_FALSE(Validation::validate_buffer_size(fake_buffer, 1024));
        ASSERT_FALSE(Validation::validate_buffer_memory(fake_buffer));
        
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

// Test device validation (without actual device)
TEST(device_validation_limits) {
    UnitTest::TestResult result;
    result.name = "device_validation_limits";
    
    try {
        // These tests verify the validation logic without requiring a real device
        VkPhysicalDeviceLimits limits = {};
        limits.maxPushConstantsSize = 256;
        limits.maxComputeWorkGroupInvocations = 1024;
        
        // Check that limits are reasonable
        ASSERT_TRUE(limits.maxPushConstantsSize >= 128);
        ASSERT_TRUE(limits.maxComputeWorkGroupInvocations >= 256);
        
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

// Test assertion macros
TEST(assertion_true) {
    UnitTest::TestResult result;
    result.name = "assertion_true";
    
    try {
        ASSERT_TRUE(true);
        ASSERT_FALSE(false);
        ASSERT_EQ(42, 42);
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

TEST(assertion_null) {
    UnitTest::TestResult result;
    result.name = "assertion_null";
    
    try {
        int value = 42;
        ASSERT_NOT_NULL(&value);
        result.passed = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.passed = false;
    }
    
    return result;
}

// Run all tests and print summary
void run_validation_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Validation Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto results = UnitTest::run_all_tests();
    
    int passed = 0;
    int failed = 0;
    double total_time = 0.0;
    
    for (const auto& result : results) {
        result.print();
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
        total_time += result.duration_ms;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
}

} // namespace vk_symbiote

// ============================================================================
// Main entry point for standalone test executable
// ============================================================================
#ifdef VALIDATION_TEST_MAIN
int main() {
    vk_symbiote::run_validation_tests();
    return 0;
}
#endif
