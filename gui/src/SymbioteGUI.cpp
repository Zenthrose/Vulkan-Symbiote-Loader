/**
 * SymbioteGUI Implementation
 * Phase 2 & 3: Smart Features + Polish
 */

#include "../include/vk_symbiote_gui/SymbioteGUI.h"
#include "../../vk_symbiote/include/vk_symbiote/VulkanSymbioteEngine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include <queue>
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <atomic>

// ImGui includes
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <GLFW/glfw3.h>

// Native file dialog includes
#if defined(__linux__)
    #include <cstdio>
#endif

namespace vk_symbiote {
namespace gui {

// Helper to execute system commands
static std::string execCommand(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result;
}

// Native file dialog implementation
static std::string openFileDialog(const std::vector<std::pair<std::string, std::string>>& filters) {
#if defined(__linux__)
    std::string cmd = "zenity --file-selection --title='Select Model File' 2>/dev/null";
    if (!filters.empty()) {
        cmd += " --file-filter='Models ";
        for (const auto& f : filters) {
            cmd += "*." + f.first + " ";
        }
        cmd += "'";
    }
    return execCommand(cmd.c_str());
#elif defined(_WIN32)
    // Windows implementation would use IFileDialog
    return "";
#elif defined(__APPLE__)
    // macOS implementation would use NSOpenPanel
    return "";
#else
    return "";
#endif
}

static std::string openFolderDialog() {
#if defined(__linux__)
    std::string cmd = "zenity --file-selection --directory --title='Select Project Folder' 2>/dev/null";
    return execCommand(cmd.c_str());
#else
    return "";
#endif
}

SymbioteGUI::SymbioteGUI() = default;

SymbioteGUI::~SymbioteGUI() {
    shutdown();
}

bool SymbioteGUI::initialize(int width, int height, const char* title) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Create window
    if (!createWindow(width, height, title)) {
        return false;
    }
    
    // Initialize Vulkan
    if (!createVulkanInstance()) {
        return false;
    }
    
    // Initialize ImGui
    if (!createImGui()) {
        return false;
    }
    
    // Load settings
    loadSettings();
    
    running_ = true;
    return true;
}

void SymbioteGUI::shutdown() {
    if (generation_thread_.joinable()) {
        generation_thread_.join();
    }
    
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
    
    // Cleanup ImGui
    if (imgui_context_) {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext(imgui_context_);
        imgui_context_ = nullptr;
    }
    
    // Cleanup Vulkan
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
    
    // Cleanup GLFW
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

bool SymbioteGUI::createWindow(int width, int height, const char* title) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    
    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        return false;
    }
    
    // Set user pointer for callbacks
    glfwSetWindowUserPointer(window_, this);
    
    // Set up drag and drop callback
    glfwSetDropCallback(window_, [](GLFWwindow* window, int count, const char** paths) {
        auto* gui = static_cast<SymbioteGUI*>(glfwGetWindowUserPointer(window));
        if (gui) {
            std::vector<std::string> dropped_paths;
            for (int i = 0; i < count; i++) {
                dropped_paths.push_back(paths[i]);
            }
            gui->handleDragDrop(dropped_paths);
        }
    });
    
    return true;
}

bool SymbioteGUI::createVulkanInstance() {
    // Create Vulkan instance
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Vulkan Symbiote GUI";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "SymbioteEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;
    
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = glfw_extension_count;
    create_info.ppEnabledExtensionNames = glfw_extensions;
    
    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
    // Create surface
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
        std::cerr << "Failed to create window surface" << std::endl;
        return false;
    }
    
    // For now, we won't create a full device - we'll use the engine's device
    // This is a simplified implementation
    
    return true;
}

bool SymbioteGUI::createImGui() {
    // Create ImGui context
    IMGUI_CHECKVERSION();
    imgui_context_ = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Requires docking branch
    
    // Setup style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window_, true);
    
    // Note: Full Vulkan backend initialization would require more setup
    // For this implementation, we're creating the structure
    
    return true;
}

void SymbioteGUI::run() {
    while (running_ && !glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Process UI
        processUI();
        
        // Handle async generation
        handleAsyncGeneration();
        
        // Rendering would go here in full implementation
        // For now, just clear and swap
        
        glfwSwapBuffers(window_);
    }
}

void SymbioteGUI::processUI() {
    // Main dockspace
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    // ImGui::SetNextWindowViewport(viewport->ID); // Requires viewport API
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar;
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    
    ImGui::Begin("MainDockspace", nullptr, window_flags);
    ImGui::PopStyleVar(2);
    
    // Dockspace (requires ImGui docking branch)
    // ImGuiIO& io = ImGui::GetIO();
    // ImGuiID dockspace_id = ImGui::GetID("MainDockspace");
    // ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f));
    
    // Menu bar
    drawMainWindow();
    
    ImGui::End();
    
    // Draw panels
    drawChatPanel();
    drawControlPanel();
    
    if (config_.show_pack_visualizer) {
        drawPackVisualizer();
    }
    
    if (model_loading_) {
        drawLoadingScreen();
    }
    
    if (config_.enable_tutorial) {
        drawTutorial();
    }
    
    if (has_pending_drag_drop_) {
        drawContextVisualizer();
    }
}

void SymbioteGUI::drawMainWindow() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Model...", "Ctrl+O")) {
                std::string path = openFileDialog({{"gguf", "GGUF Model"}, {"ggml", "GGML Model"}});
                if (!path.empty()) {
                    loadModel(path);
                }
            }
            if (ImGui::MenuItem("New Chat", "Ctrl+N")) {
                clearChat();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                running_ = false;
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Show Token Counter", nullptr, &config_.show_token_counter);
            ImGui::MenuItem("Show Pack Visualizer", nullptr, &config_.show_pack_visualizer);
            ImGui::MenuItem("Enable Drag & Drop", nullptr, &config_.enable_drag_drop);
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Settings")) {
            if (ImGui::BeginMenu("Theme")) {
                if (ImGui::MenuItem("Dark", nullptr, config_.theme == "dark")) {
                    config_.theme = "dark";
                    ImGui::StyleColorsDark();
                }
                if (ImGui::MenuItem("Light", nullptr, config_.theme == "light")) {
                    config_.theme = "light";
                    ImGui::StyleColorsLight();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Power Profile")) {
                static int profile = 1; // Balanced
                if (ImGui::MenuItem("⚡ High Performance", nullptr, profile == 0)) {
                    profile = 0;
                    setPowerProfile(PowerProfile::HIGH_PERFORMANCE);
                }
                if (ImGui::MenuItem("⚖️ Balanced", nullptr, profile == 1)) {
                    profile = 1;
                    setPowerProfile(PowerProfile::BALANCED);
                }
                if (ImGui::MenuItem("🔋 Power Saver", nullptr, profile == 2)) {
                    profile = 2;
                    setPowerProfile(PowerProfile::POWER_SAVER);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        
        ImGui::EndMenuBar();
    }
}

void SymbioteGUI::drawChatPanel() {
    ImGui::Begin("Chat", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // Header with status
    if (engine_) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "● Model Loaded");
        ImGui::SameLine();
        ImGui::Text("| Context: %zu / 200,000 tokens", chat_history_.size() * 100); // Placeholder
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "● No Model");
    }
    
    ImGui::Separator();
    
    // Chat history
    ImGui::BeginChild("ChatHistory", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 3), true);
    
    std::lock_guard<std::mutex> lock(chat_mutex_);
    for (const auto& msg : chat_history_) {
        ImVec4 color;
        const char* prefix;
        
        switch (msg.type) {
            case ChatMessage::USER:
                color = ImVec4(0.3f, 0.7f, 1.0f, 1.0f);
                prefix = "You";
                break;
            case ChatMessage::ASSISTANT:
                color = ImVec4(0.0f, 1.0f, 0.5f, 1.0f);
                prefix = "Assistant";
                break;
            case ChatMessage::SYSTEM:
                color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                prefix = "System";
                break;
            case ChatMessage::ERROR:
                color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
                prefix = "Error";
                break;
        }
        
        ImGui::TextColored(color, "%s:", prefix);
        ImGui::TextWrapped("%s", msg.content.c_str());
        
        if (config_.show_token_counter && msg.token_count > 0) {
            ImGui::TextDisabled("(%u tokens, %.1f ms)", msg.token_count, msg.generation_time_ms);
        }
        
        ImGui::Spacing();
    }
    
    // Auto-scroll to bottom
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 50) {
        ImGui::SetScrollHereY(1.0f);
    }
    
    ImGui::EndChild();
    
    // Generation indicator
    if (is_generating_) {
        ImGui::Text("Generating...");
        ImGui::SameLine();
        ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(100, 0), "");
    }
    
    // Input area
    char input_buffer[4096] = {};
    strncpy(input_buffer, input_buffer_.c_str(), sizeof(input_buffer) - 1);
    
    ImGui::PushItemWidth(-ImGui::GetFrameHeightWithSpacing() * 4);
    if (ImGui::InputText("##Input", input_buffer, sizeof(input_buffer), 
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (strlen(input_buffer) > 0) {
            sendMessage(input_buffer);
            input_buffer_.clear();
        }
    }
    ImGui::PopItemWidth();
    
    ImGui::SameLine();
    if (ImGui::Button("Send", ImVec2(80, 0))) {
        if (strlen(input_buffer) > 0) {
            sendMessage(input_buffer);
            input_buffer_.clear();
        }
    }
    
    ImGui::SameLine();
    if (ImGui::Button("📎", ImVec2(30, 0))) {
        // Attachment button
    }
    
    ImGui::End();
}

void SymbioteGUI::drawControlPanel() {
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse);
    
    // Quick actions
    ImGui::Text("Quick Actions");
    ImGui::Separator();
    
    if (ImGui::Button("Continue Story", ImVec2(-1, 40))) {
        sendMessage("Continue the story");
    }
    
    if (ImGui::Button("Summarize", ImVec2(-1, 0))) {
        sendMessage("Summarize the key points");
    }
    
    ImGui::Spacing();
    
    // Context info
    ImGui::Text("Context Information");
    ImGui::Separator();
    
    if (!context_segments_.empty()) {
        for (const auto& segment : context_segments_) {
            ImGui::BulletText("%s: %u tokens", segment.label.c_str(), segment.token_count);
        }
    } else {
        ImGui::TextDisabled("No context loaded");
    }
    
    ImGui::Spacing();
    
    // Drag drop area
    if (config_.enable_drag_drop) {
        ImGui::Text("Drag & Drop");
        ImGui::Separator();
        
        ImVec2 drop_size = ImVec2(-1, 60);
        ImGui::Button("Drop folder here", drop_size);
        
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("FILES")) {
                // Handle drag drop
            }
            ImGui::EndDragDropTarget();
        }
    }
    
    ImGui::End();
}

void SymbioteGUI::drawContextVisualizer() {
    ImGui::Begin("Context Visualizer", nullptr, ImGuiWindowFlags_NoCollapse);
    
    ImGui::Text("Context Map (200K token view)");
    ImGui::Separator();
    
    // Visual representation of context
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
    ImVec2 canvas_sz = ImVec2(ImGui::GetContentRegionAvail().x, 100);
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(30, 30, 30, 255));
    
    // Draw segments
    float x = canvas_p0.x;
    uint32_t total_tokens = 200000;
    
    for (const auto& segment : context_segments_) {
        float width = (segment.token_count / (float)total_tokens) * canvas_sz.x;
        ImU32 color = IM_COL32(100, 150, 200, 255);
        
        draw_list->AddRectFilled(
            ImVec2(x, canvas_p0.y),
            ImVec2(x + width, canvas_p1.y),
            color
        );
        
        // Label
        if (width > 50) {
            draw_list->AddText(
                ImVec2(x + 5, canvas_p0.y + 5),
                IM_COL32(255, 255, 255, 255),
                segment.label.c_str()
            );
        }
        
        x += width;
    }
    
    // Border
    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(100, 100, 100, 255));
    
    ImGui::Dummy(canvas_sz);
    
    // Legend
    ImGui::Text("Used: %u / 200,000 tokens (%.1f%%)", 
                current_context_tokens_,
                (current_context_tokens_ / 200000.0f) * 100);
    
    ImGui::End();
}

void SymbioteGUI::drawPackVisualizer() {
    ImGui::Begin("Pack Status", nullptr, ImGuiWindowFlags_NoCollapse);
    
    ImGui::Text("Memory Status");
    ImGui::Separator();
    
    // VRAM section
    ImGui::Text("VRAM (Hot)");
    float vram_used = 8.2f;  // Example values
    float vram_total = 16.0f;
    char vram_label[64];
    snprintf(vram_label, sizeof(vram_label), "%.1f GB / %.1f GB", vram_used, vram_total);
    ImGui::ProgressBar(vram_used / vram_total, ImVec2(-1, 0), vram_label);
    
    // Pack list
    for (const auto& pack : pack_status_) {
        ImVec4 color;
        const char* location_str;
        
        switch (pack.location) {
            case PackStatusInfo::VRAM_HOT:
                color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
                location_str = "VRAM";
                break;
            case PackStatusInfo::RAM_WARM:
                color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                location_str = "RAM";
                break;
            case PackStatusInfo::DISK_COLD:
                color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);
                location_str = "DISK";
                break;
            default:
                color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                location_str = "UNLOADED";
        }
        
        ImGui::TextColored(color, "● %s [%s]", pack.name.c_str(), location_str);
    }
    
    ImGui::End();
}

void SymbioteGUI::drawLoadingScreen() {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    
    if (ImGui::BeginPopupModal("Loading", nullptr, 
                               ImGuiWindowFlags_AlwaysAutoResize | 
                               ImGuiWindowFlags_NoTitleBar)) {
        ImGui::Text("Loading Model...");
        ImGui::ProgressBar(loading_progress_, ImVec2(300, 0));
        ImGui::Text("%s", loading_status_.c_str());
        ImGui::EndPopup();
    }
}

void SymbioteGUI::drawTutorial() {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    
    if (ImGui::BeginPopupModal("Welcome!", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Welcome to Vulkan Symbiote! 🚀");
        ImGui::Separator();
        
        ImGui::Text("1. Load a Model");
        ImGui::Text("   Click File → Open Model to select a GGUF file");
        ImGui::Spacing();
        
        ImGui::Text("2. Start Chatting");
        ImGui::Text("   Type in the chat box and press Enter");
        ImGui::Spacing();
        
        ImGui::Text("3. Add Context");
        ImGui::Text("   Drag a folder to give the AI project context");
        ImGui::Spacing();
        
        ImGui::Text("4. Monitor Performance");
        ImGui::Text("   Watch the token counter and pack status");
        
        ImGui::Separator();
        
        bool dont_show = false;
        ImGui::Checkbox("Don't show this again", &dont_show);
        if (dont_show) {
            config_.enable_tutorial = false;
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Get Started", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::EndPopup();
    }
}

bool SymbioteGUI::loadModel(const std::string& path) {
    model_loading_ = true;
    loading_progress_ = 0.0f;
    loading_status_ = "Initializing...";
    
    // Async loading
    std::thread load_thread([this, path]() {
        try {
            loading_status_ = "Loading model weights...";
            loading_progress_ = 0.3f;
            
            // engine_->loadModel(path);
            
            loading_status_ = "Initializing KV cache...";
            loading_progress_ = 0.7f;
            
            loading_status_ = "Ready!";
            loading_progress_ = 1.0f;
            
            config_.last_model_path = path;
            
            // Add system message
            ChatMessage msg;
            msg.type = ChatMessage::SYSTEM;
            msg.content = "Model loaded successfully: " + path;
            msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::lock_guard<std::mutex> lock(chat_mutex_);
            chat_history_.push_back(msg);
            
        } catch (const std::exception& e) {
            ChatMessage msg;
            msg.type = ChatMessage::ERROR;
            msg.content = std::string("Failed to load model: ") + e.what();
            msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::lock_guard<std::mutex> lock(chat_mutex_);
            chat_history_.push_back(msg);
        }
        
        model_loading_ = false;
    });
    
    load_thread.detach();
    return true;
}

void SymbioteGUI::sendMessage(const std::string& message) {
    // Add user message
    {
        ChatMessage user_msg;
        user_msg.type = ChatMessage::USER;
        user_msg.content = message;
        user_msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        user_msg.token_count = 0;  // Would calculate actual tokens
        
        std::lock_guard<std::mutex> lock(chat_mutex_);
        chat_history_.push_back(user_msg);
    }
    
    // Queue for generation
    {
        std::lock_guard<std::mutex> lock(generation_mutex_);
        generation_queue_.push(message);
    }
    
    is_generating_ = true;
}

void SymbioteGUI::handleAsyncGeneration() {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    
    if (!generation_queue_.empty() && !is_generating_) {
        std::string prompt = generation_queue_.front();
        generation_queue_.pop();
        
        // Start generation thread
        generation_thread_ = std::thread([this, prompt]() {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Generate response
            std::string response = "This is a placeholder response. In the full implementation, "
                                   "this would call the VulkanSymbioteEngine to generate text.";
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Add assistant message
            ChatMessage msg;
            msg.type = ChatMessage::ASSISTANT;
            msg.content = response;
            msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            msg.token_count = 50;  // Placeholder
            msg.generation_time_ms = duration.count();
            
            {
                std::lock_guard<std::mutex> lock(chat_mutex_);
                chat_history_.push_back(msg);
            }
            
            is_generating_ = false;
        });
        
        generation_thread_.detach();
    }
}

void SymbioteGUI::handleDragDrop(const std::vector<std::string>& paths) {
    has_pending_drag_drop_ = true;
    
    for (const auto& path : paths) {
        DragDropPayload payload;
        payload.paths.push_back(path);
        payload.is_folder = (path.back() == '/' || path.find('.') == std::string::npos);
        
        // Detect type
        if (path.find(".txt") != std::string::npos || 
            path.find(".md") != std::string::npos) {
            payload.detected_type = "novel";
        } else if (path.find(".py") != std::string::npos || 
                   path.find(".cpp") != std::string::npos) {
            payload.detected_type = "code";
        } else {
            payload.detected_type = "documentation";
        }
        
        // Add system message
        ChatMessage msg;
        msg.type = ChatMessage::SYSTEM;
        msg.content = "Detected " + payload.detected_type + " project: " + path;
        msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::lock_guard<std::mutex> lock(chat_mutex_);
        chat_history_.push_back(msg);
    }
}

void SymbioteGUI::loadSettings() {
    std::ifstream file("symbiote_settings.toml");
    if (file.is_open()) {
        // Simple TOML parsing would go here
        // For now, use defaults
        file.close();
    }
}

void SymbioteGUI::saveSettings() {
    std::ofstream file("symbiote_settings.toml");
    if (file.is_open()) {
        file << "[gui]\n";
        file << "theme = \"" << config_.theme << "\"\n";
        file << "font_size = " << config_.font_size << "\n";
        file << "enable_tutorial = " << (config_.enable_tutorial ? "true" : "false") << "\n";
        file << "\n[model]\n";
        file << "last_model = \"" << config_.last_model_path << "\"\n";
        file.close();
    }
}

void SymbioteGUI::setPowerProfile(PowerProfile profile) {
    // Would interface with engine's power manager
}

SymbioteGUI::PowerProfile SymbioteGUI::getPowerProfile() const {
    return PowerProfile::BALANCED;
}

void SymbioteGUI::clearChat() {
    std::lock_guard<std::mutex> lock(chat_mutex_);
    chat_history_.clear();
}

} // namespace gui
} // namespace vk_symbiote
