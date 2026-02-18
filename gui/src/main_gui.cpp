/**
 * Symbiote GUI - User-friendly interface for Vulkan Symbiote
 * 
 * Phase 1 Implementation:
 * - Basic window with ImGui + GLFW + Vulkan
 * - Model file picker dialog
 * - Chat interface
 * - Real-time token counter
 * 
 * Usage:
 *   ./symbiote_chat [optional_model_path]
 * 
 * Features:
 *   - Double-click to launch (no terminal required)
 *   - Drag and drop folders for context
 *   - Visual feedback during loading
 *   - Settings persistence
 */

#include "vk_symbiote_gui/SymbioteGUI.h"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    // Set error callback for GLFW
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "GLFW Error " << error << ": " << description << std::endl;
    });
    
    // Create GUI instance
    auto gui = std::make_unique<vk_symbiote::gui::SymbioteGUI>();
    
    // Initialize GUI (1600x900 window)
    if (!gui->initialize(1600, 900, "Vulkan Symbiote - AI Chat")) {
        std::cerr << "Failed to initialize GUI" << std::endl;
        return 1;
    }
    
    // If model path provided as argument, load it
    if (argc > 1) {
        std::cout << "Loading model from command line: " << argv[1] << std::endl;
        gui->loadModel(argv[1]);
    }
    
    // Run main loop
    gui->run();
    
    // Cleanup
    gui->shutdown();
    
    return 0;
}
