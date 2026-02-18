cmake_minimum_required(VERSION 3.20)
project(SymbioteGUI VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(PkgConfig REQUIRED)

# Find ImGui
# Try to find system ImGui first, otherwise use bundled version
find_path(IMGUI_INCLUDE_DIR imgui.h 
    PATHS /usr/include /usr/local/include ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
)

if(NOT IMGUI_INCLUDE_DIR)
    message(STATUS "ImGui not found system-wide, will use bundled version")
    set(IMGUI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
endif()

# ImGui implementation files
set(IMGUI_SOURCES
    ${IMGUI_INCLUDE_DIR}/imgui.cpp
    ${IMGUI_INCLUDE_DIR}/imgui_draw.cpp
    ${IMGUI_INCLUDE_DIR}/imgui_tables.cpp
    ${IMGUI_INCLUDE_DIR}/imgui_widgets.cpp
    ${IMGUI_INCLUDE_DIR}/imgui_demo.cpp
    ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_INCLUDE_DIR}/backends/imgui_impl_vulkan.cpp
)

# Native file dialog
# Linux: use zenity or kdialog
# macOS: use native dialogs
# Windows: use IFileDialog
set(NFD_SOURCES
    src/NativeFileDialog.cpp
)

# GUI library
add_library(symbiote_gui STATIC
    src/SymbioteGUI.cpp
    src/ChatSession.cpp
    ${IMGUI_SOURCES}
    ${NFD_SOURCES}
)

target_include_directories(symbiote_gui PUBLIC
    include
    ${IMGUI_INCLUDE_DIR}
    ${IMGUI_INCLUDE_DIR}/backends
    ../vk_symbiote/include
)

target_link_libraries(symbiote_gui PUBLIC
    Vulkan::Vulkan
    glfw
    vk_symbiote
)

target_compile_definitions(symbiote_gui PUBLIC
    IMGUI_IMPL_VULKAN_USE_VOLK=0
)

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
