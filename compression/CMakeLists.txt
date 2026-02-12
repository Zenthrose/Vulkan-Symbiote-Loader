cmake_minimum_required(VERSION 3.16)
project(CompressionBackends LANGUAGES CXX)

add_library(blosc2_backend STATIC
    include/Blosc2Compression.h
    src/Blosc2Compression.cpp
)

target_include_directories(blosc2_backend PUBLIC include)

find_package(Blosc2 QUIET)
if(Blosc2_FOUND)
    target_compile_definitions(blosc2_backend PRIVATE BLOSC2_ENABLED)
    target_link_libraries(blosc2_backend PRIVATE Blosc2::Blosc2)
endif()

set_target_properties(blosc2_backend PROPERTIES
    FOLDER "Compression"
)
