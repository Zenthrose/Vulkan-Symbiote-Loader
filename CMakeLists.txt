cmake_minimum_required(VERSION 3.16)
project(VulkanSymbioteLoader VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build test suite" OFF)
option(BUILD_GUI "Build GUI application (requires ImGui + GLFW)" ON)
option(ENABLE_ZFP "Enable ZFP compression support" ON)
option(ENABLE_BLOSC2 "Enable Blosc2 compression support" ON)

# Find dependencies
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found. Please install Vulkan SDK.")
endif()

# Check Vulkan version
message(STATUS "Found Vulkan: ${Vulkan_VERSION}")
if(Vulkan_VERSION VERSION_LESS "1.2")
    message(FATAL_ERROR "Vulkan 1.2+ required, found ${Vulkan_VERSION}")
endif()

# Add subdirectories
add_subdirectory(compression)
add_subdirectory(vk_symbiote)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_GUI)
    # Check for GUI dependencies
    find_package(glfw3 QUIET)
    if(glfw3_FOUND)
        message(STATUS "✅ GUI dependencies found (GLFW)")
        add_subdirectory(gui)
    else()
        message(WARNING "⚠️  GLFW not found. GUI application will not be built.")
        message(STATUS "   Install GLFW: sudo apt-get install libglfw3-dev")
    endif()
endif()

# Summary
message(STATUS "")
message(STATUS "=== Vulkan Symbiote Loader Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Vulkan: ${Vulkan_VERSION}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "ZFP compression: ${ENABLE_ZFP}")
message(STATUS "Blosc2 compression: ${ENABLE_BLOSC2}")
message(STATUS "============================================")
message(STATUS "")
