#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <queue>

// Forward declarations
struct ImGuiContext;

namespace vk_symbiote {
class VulkanSymbioteEngine;

namespace gui {

// Chat message (defined before ChatSession)
struct ChatMessage {
    enum Type { USER, ASSISTANT, SYSTEM, ERROR };
    Type type;
    std::string content;
    uint64_t timestamp;
    uint32_t token_count;
    float generation_time_ms;
};

// Chat session for managing conversation
class ChatSession {
public:
    struct ContextSegment {
        std::string label;
        uint32_t token_count;
        uint64_t timestamp;
    };

    ChatSession();
    ~ChatSession();

    void addMessage(const ChatMessage& message);
    std::vector<ChatMessage> getMessages() const;
    void clear();

    uint32_t getTokenCount() const;
    uint32_t getMaxTokens() const;
    void setMaxTokens(uint32_t max_tokens);

    void addContextSegment(const std::string& label, uint32_t token_count);
    std::vector<ContextSegment> getContextSegments() const;

    void loadContextFromFiles(const std::vector<std::string>& files);
    std::string buildPrompt(const std::string& user_input) const;

private:
    std::vector<ChatMessage> messages_;
    mutable std::mutex messages_mutex_;
    uint32_t current_tokens_ = 0;
    uint32_t max_context_tokens_ = 200000;
    std::vector<ContextSegment> context_segments_;
};

// GUI Configuration
struct GUIConfig {
    bool enable_tutorial = true;
    bool auto_load_last_model = false;
    std::string last_model_path;
    std::string theme = "dark";  // dark, light, system
    int font_size = 16;
    bool show_token_counter = true;
    bool show_pack_visualizer = true;
    bool enable_drag_drop = true;
};

// Drag-drop payload for files/folders
struct DragDropPayload {
    std::vector<std::string> paths;
    bool is_folder;
    std::string detected_type;  // "novel", "code", "documentation", etc.
};

// Context visualization data
struct ContextVisualization {
    uint32_t total_tokens;
    uint32_t used_tokens;
    uint32_t cached_tokens;
    std::vector<std::pair<uint32_t, std::string>> segments;  // (token_count, label)
};

// Pack status for visualization
struct PackStatusInfo {
    uint64_t pack_id;
    std::string name;
    enum Location { VRAM_HOT, RAM_WARM, DISK_COLD, UNLOADED } location;
    float priority;
    uint64_t last_access;
    size_t size_bytes;
};

class SymbioteGUI {
public:
    SymbioteGUI();
    ~SymbioteGUI();

    // Initialize GUI with Vulkan
    bool initialize(int width = 1600, int height = 900, const char* title = "Vulkan Symbiote");
    void shutdown();

    // Main loop
    void run();

    // Model loading
    bool loadModel(const std::string& path);
    void unloadModel();
    bool isModelLoaded() const;

    // Chat interface
    void sendMessage(const std::string& message);
    void clearChat();
    const std::vector<ChatMessage>& getChatHistory() const;

    // Drag-drop handling
    void handleDragDrop(const std::vector<std::string>& paths);
    
    // Context management
    void loadContextFromFolder(const std::string& folder_path);
    void clearContext();
    ContextVisualization getContextVisualization() const;

    // Power profile
    enum PowerProfile { HIGH_PERFORMANCE, BALANCED, POWER_SAVER };
    void setPowerProfile(PowerProfile profile);
    PowerProfile getPowerProfile() const;

    // Visualization
    std::vector<PackStatusInfo> getPackStatus() const;
    std::string getVitalityOraclePrediction() const;

    // Settings
    void saveSettings();
    void loadSettings();
    GUIConfig& getConfig() { return config_; }

private:
    // Window and Vulkan
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = nullptr;
    
    // ImGui
    ImGuiContext* imgui_context_ = nullptr;
    VkDescriptorPool imgui_descriptor_pool_ = VK_NULL_HANDLE;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkImageView> swapchain_image_views_;
    std::vector<VkCommandBuffer> command_buffers_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    
    // Engine
    std::unique_ptr<VulkanSymbioteEngine> engine_;
    std::unique_ptr<ChatSession> chat_session_;
    
    // State
    GUIConfig config_;
    bool running_ = false;
    bool model_loading_ = false;
    float loading_progress_ = 0.0f;
    std::string loading_status_;
    
    // Chat
    std::vector<ChatMessage> chat_history_;
    std::mutex chat_mutex_;
    std::string input_buffer_;
    std::vector<ChatSession::ContextSegment> context_segments_;
    uint32_t current_context_tokens_ = 0;
    std::vector<PackStatusInfo> pack_status_;
    
    // Async generation
    std::thread generation_thread_;
    std::queue<std::string> generation_queue_;
    std::mutex generation_mutex_;
    bool is_generating_ = false;
    
    // Drag-drop
    DragDropPayload pending_payload_;
    bool has_pending_drag_drop_ = false;
    
    // Rendering
    uint32_t current_frame_ = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    
    // Private methods
    bool createWindow(int width, int height, const char* title);
    bool createVulkanInstance();
    bool createVulkanDevice();
    bool createSwapchain();
    bool createImGui();
    void renderFrame();
    void processUI();
    void drawMainWindow();
    void drawChatPanel();
    void drawControlPanel();
    void drawContextVisualizer();
    void drawPackVisualizer();
    void drawSettingsWindow();
    void drawTutorial();
    void drawLoadingScreen();
    void handleAsyncGeneration();
    void loadSettingsFromFile(const std::string& path);
    void saveSettingsToFile(const std::string& path);
};

} // namespace gui
} // namespace vk_symbiote
