cmake_minimum_required(VERSION 3.16)
project(CompressionBackends LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and link Blosc2
set(BLOSC2_ENABLED FALSE)
find_package(Blosc2 QUIET)
if(Blosc2_FOUND)
    set(BLOSC2_ENABLED TRUE)
    message(STATUS "✅ Blosc2 compression enabled")
else()
    # Try to find blosc2 via pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(BLOSC2 blosc2)
        if(BLOSC2_FOUND)
            set(BLOSC2_ENABLED TRUE)
            message(STATUS "✅ Blosc2 compression enabled (via pkg-config)")
        endif()
    endif()
endif()

# Find and link ZFP
set(ZFP_ENABLED FALSE)
# Skip find_package(ZFP) as it requires OpenMP which may not be properly detected
# Instead, manually find ZFP in standard locations
find_path(ZFP_INCLUDE_DIR zfp.h PATHS /usr/include /usr/local/include PATH_SUFFIXES zfp)
find_library(ZFP_LIBRARY zfp PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

if(ZFP_INCLUDE_DIR AND ZFP_LIBRARY)
    set(ZFP_ENABLED TRUE)
    message(STATUS "✅ ZFP compression enabled (found at ${ZFP_LIBRARY})")
else()
    message(STATUS "⚠️  ZFP not found - lossy compression disabled")
endif()

# Source files - only include implementation files when libraries are found
set(COMPRESSION_SOURCES)

if(BLOSC2_ENABLED)
    list(APPEND COMPRESSION_SOURCES src/Blosc2Compression.cpp)
endif()

if(ZFP_ENABLED)
    list(APPEND COMPRESSION_SOURCES src/ZFPCompression.cpp)
endif()

if(BLOSC2_ENABLED AND ZFP_ENABLED)
    list(APPEND COMPRESSION_SOURCES src/HybridCompression.cpp)
endif()

# Need at least one source file for the library
if(NOT COMPRESSION_SOURCES)
    # Create a minimal source file when no backends are enabled
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/noop.cpp" 
        "// No compression backends enabled - headers provide fallback stubs\n")
    list(APPEND COMPRESSION_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/noop.cpp")
endif()

set(COMPRESSION_HEADERS
    include/CompressionBackend.h
    include/Blosc2Compression.h
    include/ZFPCompression.h
    include/HybridCompression.h
)

add_library(compression_backends STATIC ${COMPRESSION_SOURCES} ${COMPRESSION_HEADERS})

target_include_directories(compression_backends PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BLOSC2_ENABLED)
    target_compile_definitions(compression_backends PRIVATE BLOSC2_ENABLED)
    if(TARGET Blosc2::Blosc2)
        target_link_libraries(compression_backends PUBLIC Blosc2::Blosc2)
    else()
        target_include_directories(compression_backends PRIVATE ${BLOSC2_INCLUDE_DIRS})
        target_link_libraries(compression_backends PUBLIC ${BLOSC2_LIBRARIES})
    endif()
endif()

if(ZFP_ENABLED)
    target_compile_definitions(compression_backends PRIVATE ZFP_ENABLED)
    if(TARGET zfp::zfp)
        target_link_libraries(compression_backends PUBLIC zfp::zfp)
    else()
        target_include_directories(compression_backends PRIVATE ${ZFP_INCLUDE_DIR})
        target_link_libraries(compression_backends PUBLIC ${ZFP_LIBRARY})
    endif()
endif()

# Set folder property for IDEs
set_target_properties(compression_backends PROPERTIES
    FOLDER "Compression"
)

# Export configuration
install(TARGETS compression_backends
    EXPORT CompressionBackendsConfig
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "=== Compression Backends Configuration ===")
message(STATUS "Blosc2: ${BLOSC2_ENABLED}")
message(STATUS "ZFP: ${ZFP_ENABLED}")
if(BLOSC2_ENABLED AND ZFP_ENABLED)
    message(STATUS "Hybrid: ON")
else()
    message(STATUS "Hybrid: OFF (requires both)")
endif()
message(STATUS "===========================================")
