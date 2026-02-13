cmake_minimum_required(VERSION 3.16)
project(VulkanSymbiote VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(VULKAN_1_2_FALLBACK "Enable Vulkan 1.2 fallback mode (no timeline semaphores)" OFF)
option(ENABLE_TIMELINE_SEMAPHORES "Enable timeline semaphores for async operations" ON)
option(ENABLE_ARM_NEON "Enable ARM NEON optimizations" OFF)
option(ENABLE_ERROR_RETRY "Enable retry on partial reads/corrupt tensors" ON)

# Compiler flags based on architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "🔧 ARM64 architecture detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd")
    set(ENABLE_ARM_NEON ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armhf")
    message(STATUS "🔧 ARMv7 architecture detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard")
    set(ENABLE_ARM_NEON ON)
else()
    message(STATUS "🔧 x86_64 architecture detected")
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# Vulkan 1.2 fallback
if(VULKAN_1_2_FALLBACK)
    message(STATUS "⚠️  Vulkan 1.2 fallback mode enabled")
    add_compile_definitions(VULKAN_1_2_FALLBACK=1)
    set(ENABLE_TIMELINE_SEMAPHORES OFF)
else()
    message(STATUS "✅ Vulkan 1.3+ with timeline semaphores enabled")
endif()

# Error retry support
if(ENABLE_ERROR_RETRY)
    message(STATUS "✅ Error retry enabled")
    add_compile_definitions(ENABLE_ERROR_RETRY=1)
endif()

# ARM NEON
if(ENABLE_ARM_NEON)
    message(STATUS "✅ ARM NEON optimizations enabled")
    add_compile_definitions(ENABLE_ARM_NEON=1)
endif()

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found")
endif()

# Find Vulkan Memory Allocator header
find_path(VMA_INCLUDE_DIRS
    NAMES vk_mem_alloc.h
    PATHS /usr/include /usr/local/include
    DOC "Vulkan Memory Allocator include directory"
)

if(NOT VMA_INCLUDE_DIRS)
    message(FATAL_ERROR "Vulkan Memory Allocator header vk_mem_alloc.h not found in /usr/include or /usr/local/include")
endif()

# Set VMA libraries as empty since it's header-only
set(VMA_LIBRARIES "")

message(STATUS "Found VMA header: ${VMA_INCLUDE_DIRS}/vk_mem_alloc.h")

# Find compression backends
add_subdirectory(../compression ${CMAKE_BINARY_DIR}/compression)

# Find ZFP for hybrid compression
if(ENABLE_ZFP)
    find_package(ZFP QUIET)
    if(NOT ZFP_FOUND)
        find_path(ZFP_INCLUDE_DIR zfp.h PATHS /usr/include /usr/local/include PATH_SUFFIXES zfp)
        find_library(ZFP_LIBRARY zfp PATHS /usr/lib /usr/local/lib)
        if(ZFP_INCLUDE_DIR AND ZFP_LIBRARY)
            set(ZFP_FOUND TRUE)
            message(STATUS "✅ ZFP found for hybrid compression")
        endif()
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/vk_symbiote)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../compression/include)
include_directories(${Vulkan_INCLUDE_DIRS})

set(ENGINE_SOURCES
    src/VulkanSymbioteEngine.cpp
    src/NomadPack.cpp
    src/VitalityOracle.cpp
    src/ShaderRuntime.cpp
    src/Tokenizer.cpp
    src/Compression.cpp
    src/Utils.cpp
    src/ConfigManager.cpp
    src/GGUFLoader.cpp
)

set(ENGINE_HEADERS
    include/vk_symbiote/VulkanSymbioteEngine.h
    include/vk_symbiote/NomadPack.h
    include/vk_symbiote/VitalityOracle.h
    include/vk_symbiote/ShaderRuntime.h
    include/vk_symbiote/Tokenizer.h
    include/vk_symbiote/Common.h
    include/vk_symbiote/GGUFLoader.h
    include/vk_symbiote/Utils.h
    include/vk_symbiote/ConfigManager.h
)

add_library(vk_symbiote STATIC ${ENGINE_SOURCES} ${ENGINE_HEADERS})
target_include_directories(vk_symbiote PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(vk_symbiote SYSTEM PRIVATE ${Vulkan_INCLUDE_DIRS} ${VMA_INCLUDE_DIRS})
target_link_libraries(vk_symbiote ${Vulkan_LIBRARIES} ${VMA_LIBRARIES} compression_backends)

# Add ZFP if found
if(ZFP_FOUND)
    target_compile_definitions(vk_symbiote PRIVATE ZFP_ENABLED)
    if(TARGET zfp::zfp)
        target_link_libraries(vk_symbiote zfp::zfp)
    else()
        target_include_directories(vk_symbiote PRIVATE ${ZFP_INCLUDE_DIR})
        target_link_libraries(vk_symbiote ${ZFP_LIBRARY})
    endif()
endif()

if(WIN32)
    target_link_libraries(vk_symbiote user32 gdi32 shell32)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(vk_symbiote pthread dl)
endif()

add_executable(vk_symbiote_example examples/basic_inference.cpp)
target_link_libraries(vk_symbiote_example vk_symbiote)

add_executable(vk_symbiote_benchmark examples/benchmark.cpp)
target_link_libraries(vk_symbiote_benchmark vk_symbiote)

message(STATUS "=== Vulkan Symbiote Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "Vulkan version: ${VULKAN_VERSION}")
message(STATUS "Timeline semaphores: ${ENABLE_TIMELINE_SEMAPHORES}")
message(STATUS "ARM NEON: ${ENABLE_ARM_NEON}")
message(STATUS "Error retry: ${ENABLE_ERROR_RETRY}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "==========================================")
