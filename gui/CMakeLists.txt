cmake_minimum_required(VERSION 3.20)
project(SymbioteGUI VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Vulkan REQUIRED)

# Check for GLFW
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    message(WARNING "GLFW3 not found - GUI will not be built")
    return()
endif()

# Find ImGui
# Try pkg-config first
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(IMGUI imgui)
endif()

# Try to find system ImGui
if(NOT IMGUI_FOUND)
    find_path(IMGUI_INCLUDE_DIR imgui.h 
        PATHS /usr/include /usr/local/include /usr/include/imgui
    )
    if(IMGUI_INCLUDE_DIR)
        set(IMGUI_FOUND TRUE)
        message(STATUS "Found system ImGui: ${IMGUI_INCLUDE_DIR}")
    endif()
endif()

# Check for bundled ImGui
if(NOT IMGUI_FOUND)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui.h)
        message(STATUS "Using bundled ImGui")
        set(IMGUI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
        set(IMGUI_FOUND TRUE)
        set(IMGUI_SOURCES
            ${IMGUI_INCLUDE_DIR}/imgui.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_draw.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_tables.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_widgets.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_demo.cpp
            ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_glfw.cpp
            ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_vulkan.cpp
        )
    else()
        message(WARNING "ImGui not found - GUI will be built without ImGui integration")
        set(IMGUI_INCLUDE_DIR "")
        set(IMGUI_SOURCES "")
    endif()
else()
    message(STATUS "Using system ImGui")
endif()

# Native file dialog
set(NFD_SOURCES
    src/NativeFileDialog.cpp
)

# GUI library
add_library(symbiote_gui STATIC
    src/SymbioteGUI.cpp
    src/ChatSession.cpp
    ${NFD_SOURCES}
)

target_include_directories(symbiote_gui PUBLIC
    include
    ../vk_symbiote/include
)

# Add ImGui includes from pkg-config if available
if(IMGUI_FOUND AND IMGUI_INCLUDE_DIRS)
    target_include_directories(symbiote_gui PUBLIC ${IMGUI_INCLUDE_DIRS})
    # Also add backends subdirectory
    target_include_directories(symbiote_gui PUBLIC /usr/include/imgui/backends)
elseif(IMGUI_INCLUDE_DIR)
    target_include_directories(symbiote_gui PUBLIC 
        ${IMGUI_INCLUDE_DIR}
        ${IMGUI_INCLUDE_DIR}/backends
    )
endif()

target_link_libraries(symbiote_gui PUBLIC
    Vulkan::Vulkan
    glfw
    vk_symbiote
    imgui
    stb
)

if(IMGUI_INCLUDE_DIR)
    target_compile_definitions(symbiote_gui PUBLIC
        IMGUI_IMPL_VULKAN_USE_VOLK=0
        HAS_IMGUI=1
    )
else()
    target_compile_definitions(symbiote_gui PUBLIC
        HAS_IMGUI=0
    )
endif()

# GUI executable
add_executable(symbiote_chat
    src/main_gui.cpp
)

target_link_libraries(symbiote_chat PRIVATE
    symbiote_gui
)

# Installation
install(TARGETS symbiote_chat
    RUNTIME DESTINATION bin
)

install(DIRECTORY assets/
    DESTINATION share/symbiote_gui/assets
    OPTIONAL
)
