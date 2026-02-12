cmake_minimum_required(VERSION 3.16)
project(VulkanSymbiote VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
target_link_libraries(vk_symbiote ${Vulkan_LIBRARIES} ${VMA_LIBRARIES})

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
message(STATUS "==========================================")
