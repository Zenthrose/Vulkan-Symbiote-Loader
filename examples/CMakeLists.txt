# Examples CMakeLists.txt

# Benchmark example
add_executable(benchmark_example
    benchmark.cpp
)

# Set include directories
get_target_property(VK_SYMBIOTE_INCLUDES vk_symbiote INCLUDE_DIRECTORIES)

# Link against vk_symbiote library
target_link_libraries(benchmark_example PRIVATE
    vk_symbiote
    compression_backends
    ${Vulkan_LIBRARIES}
)

# Include directories
target_include_directories(benchmark_example PRIVATE
    ${CMAKE_SOURCE_DIR}/vk_symbiote/include
    ${CMAKE_SOURCE_DIR}/compression/include
    ${Vulkan_INCLUDE_DIRS}
)

# Installation
install(TARGETS benchmark_example
    RUNTIME DESTINATION bin
)
