cmake_minimum_required(VERSION 3.16)
project(CompressionBackends LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable ZFP compression
option(ENABLE_ZFP "Enable ZFP compression support" ON)
option(ENABLE_BLOSC2 "Enable Blosc2 compression support" ON)

# Source files for compression backends
set(COMPRESSION_SOURCES
    src/Blosc2Compression.cpp
)

set(COMPRESSION_HEADERS
    include/CompressionBackend.h
    include/Blosc2Compression.h
    include/ZFPCompression.h
    include/HybridCompression.h
)

if(ENABLE_ZFP)
    list(APPEND COMPRESSION_SOURCES src/ZFPCompression.cpp)
endif()

if(ENABLE_ZFP AND ENABLE_BLOSC2)
    list(APPEND COMPRESSION_SOURCES src/HybridCompression.cpp)
endif()

add_library(compression_backends STATIC ${COMPRESSION_SOURCES} ${COMPRESSION_HEADERS})

target_include_directories(compression_backends PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find and link Blosc2
if(ENABLE_BLOSC2)
    find_package(Blosc2 QUIET)
    if(Blosc2_FOUND)
        target_compile_definitions(compression_backends PRIVATE BLOSC2_ENABLED)
        target_link_libraries(compression_backends PUBLIC Blosc2::Blosc2)
        message(STATUS "✅ Blosc2 compression enabled")
    else()
        # Try to find blosc2 via pkg-config
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(BLOSC2 blosc2)
            if(BLOSC2_FOUND)
                target_compile_definitions(compression_backends PRIVATE BLOSC2_ENABLED)
                target_include_directories(compression_backends PRIVATE ${BLOSC2_INCLUDE_DIRS})
                target_link_libraries(compression_backends PUBLIC ${BLOSC2_LIBRARIES})
                message(STATUS "✅ Blosc2 compression enabled (via pkg-config)")
            endif()
        endif()
    endif()
endif()

# Find and link ZFP
if(ENABLE_ZFP)
    find_package(ZFP QUIET)
    if(ZFP_FOUND)
        target_compile_definitions(compression_backends PRIVATE ZFP_ENABLED)
        target_link_libraries(compression_backends PUBLIC zfp::zfp)
        message(STATUS "✅ ZFP compression enabled")
    else()
        # Try to find ZFP in standard locations
        find_path(ZFP_INCLUDE_DIR zfp.h
            PATHS /usr/include /usr/local/include
            PATH_SUFFIXES zfp
        )
        find_library(ZFP_LIBRARY zfp
            PATHS /usr/lib /usr/local/lib
        )
        
        if(ZFP_INCLUDE_DIR AND ZFP_LIBRARY)
            target_compile_definitions(compression_backends PRIVATE ZFP_ENABLED)
            target_include_directories(compression_backends PRIVATE ${ZFP_INCLUDE_DIR})
            target_link_libraries(compression_backends PUBLIC ${ZFP_LIBRARY})
            message(STATUS "✅ ZFP compression enabled (found at ${ZFP_LIBRARY})")
        else()
            message(STATUS "⚠️  ZFP not found - lossy compression disabled")
        endif()
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
message(STATUS "Blosc2: ${ENABLE_BLOSC2}")
message(STATUS "ZFP: ${ENABLE_ZFP}")
message(STATUS "Hybrid: ${ENABLE_ZFP_AND_ENABLE_BLOSC2}")
message(STATUS "===========================================")
