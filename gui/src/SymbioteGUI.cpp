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

// Custom deleter to avoid GCC warning about function attributes
struct PipeDeleter {
    void operator()(FILE* pipe) const {
        if (pipe) pclose(pipe);
    }
};

// Helper to execute system commands
static std::string execCommand(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, PipeDeleter> pipe(popen(cmd, "r"));
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
    
    // Create chat session (engine created when model is loaded)
    chat_session_ = std::make_unique<ChatSession>();
    
    ChatMessage msg;
    msg.type = ChatMessage::SYSTEM;
    msg.content = "Welcome to Vulkan Symbiote! Open a model to start chatting.";
    msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    chat_history_.push_back(msg);
    
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
    if (imgui_descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);
    }
    
    for (auto framebuffer : framebuffers_) {
        if (framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
    }
    framebuffers_.clear();
    
    for (auto image_view : swapchain_image_views_) {
        if (image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, image_view, nullptr);
        }
    }
    swapchain_image_views_.clear();
    
    if (render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_, render_pass_, nullptr);
    }
    
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
        allocator_ = nullptr;
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
    
    // Create device and swapchain for rendering
    if (!createVulkanDevice()) {
        return false;
    }
    
    if (!createSwapchain()) {
        return false;
    }
    
    return true;
}

bool SymbioteGUI::createVulkanDevice() {
    // Select physical device
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    
    // Select first suitable device
    physical_device_ = devices[0];
    
    // Find graphics queue family
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());
    
    uint32_t graphics_family = UINT32_MAX;
    uint32_t compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &present_support);
            if (present_support) {
                graphics_family = i;
            }
        }
        // Find a compute queue (can be same as graphics)
        if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && compute_family == UINT32_MAX) {
            compute_family = i;
        }
    }
    
    // Fallback: use graphics family for compute if no separate compute queue
    if (compute_family == UINT32_MAX) {
        compute_family = graphics_family;
    }
    
    if (graphics_family == UINT32_MAX) {
        std::cerr << "Failed to find suitable queue family" << std::endl;
        return false;
    }
    
    // Create logical device with both graphics and compute queues
    std::array<VkDeviceQueueCreateInfo, 2> queue_create_infos = {};
    float queue_priority = 1.0f;
    
    // Graphics queue
    queue_create_infos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_infos[0].queueFamilyIndex = graphics_family;
    queue_create_infos[0].queueCount = 1;
    queue_create_infos[0].pQueuePriorities = &queue_priority;
    
    // Compute queue (if different from graphics)
    uint32_t queue_count = 1;
    if (compute_family != graphics_family) {
        queue_create_infos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_infos[1].queueFamilyIndex = compute_family;
        queue_create_infos[1].queueCount = 1;
        queue_create_infos[1].pQueuePriorities = &queue_priority;
        queue_count = 2;
    }
    
    const char* device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    
    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = queue_count;
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.enabledExtensionCount = 1;
    device_create_info.ppEnabledExtensionNames = device_extensions;
    
    if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device" << std::endl;
        return false;
    }
    
    vkGetDeviceQueue(device_, graphics_family, 0, &graphics_queue_);
    vkGetDeviceQueue(device_, compute_family, 0, &compute_queue_);
    
    // Create VMA allocator
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physical_device_;
    allocator_info.device = device_;
    if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) {
        std::cerr << "Failed to create VMA allocator" << std::endl;
        return false;
    }
    
    // Create command pool
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = graphics_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }
    
    return true;
}

bool SymbioteGUI::createSwapchain() {
    // Get surface capabilities
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);
    
    // Choose surface format
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data());
    
    VkSurfaceFormatKHR surface_format = formats[0];
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surface_format = format;
            break;
        }
    }
    
    // Choose present mode
    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &present_mode_count, nullptr);
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_, &present_mode_count, present_modes.data());
    
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (const auto& mode : present_modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = mode;
            break;
        }
    }
    
    // Create swapchain
    VkSwapchainCreateInfoKHR swapchain_info = {};
    swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.surface = surface_;
    swapchain_info.minImageCount = 2;
    swapchain_info.imageFormat = surface_format.format;
    swapchain_info.imageColorSpace = surface_format.colorSpace;
    swapchain_info.imageExtent = caps.currentExtent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.preTransform = caps.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = present_mode;
    swapchain_info.clipped = VK_TRUE;
    
    if (vkCreateSwapchainKHR(device_, &swapchain_info, nullptr, &swapchain_) != VK_SUCCESS) {
        std::cerr << "Failed to create swapchain" << std::endl;
        return false;
    }
    
    // Get swapchain images
    uint32_t image_count;
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    std::vector<VkImage> swapchain_images(image_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images.data());
    
    // Create render pass
    VkAttachmentDescription color_attachment = {};
    color_attachment.format = surface_format.format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    
    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    
    if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS) {
        std::cerr << "Failed to create render pass" << std::endl;
        return false;
    }
    
    // Create image views and framebuffers
    swapchain_image_views_.resize(image_count);
    framebuffers_.resize(image_count);
    for (size_t i = 0; i < image_count; i++) {
        VkImageViewCreateInfo view_info = {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = swapchain_images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = surface_format.format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.layerCount = 1;
        
        if (vkCreateImageView(device_, &view_info, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create image view" << std::endl;
            return false;
        }
        
        VkFramebufferCreateInfo fb_info = {};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = render_pass_;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &swapchain_image_views_[i];
        fb_info.width = caps.currentExtent.width;
        fb_info.height = caps.currentExtent.height;
        fb_info.layers = 1;
        
        if (vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create framebuffer" << std::endl;
            return false;
        }
    }
    
    // Allocate command buffers
    command_buffers_.resize(image_count);
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());
    
    if (vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffers" << std::endl;
        return false;
    }
    
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
    
    // Create descriptor pool for ImGui
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * 11;
    pool_info.poolSizeCount = 11;
    pool_info.pPoolSizes = pool_sizes;
    
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &imgui_descriptor_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create ImGui descriptor pool" << std::endl;
        return false;
    }
    
    // Initialize ImGui Vulkan backend
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance_;
    init_info.PhysicalDevice = physical_device_;
    init_info.Device = device_;
    init_info.QueueFamily = 0; // We know we use family 0 from createVulkanDevice
    init_info.Queue = graphics_queue_;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imgui_descriptor_pool_;
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(framebuffers_.size());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = nullptr;
    
    if (!ImGui_ImplVulkan_Init(&init_info, render_pass_)) {
        std::cerr << "Failed to initialize ImGui Vulkan backend" << std::endl;
        return false;
    }
    
    // Upload fonts
    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        std::cerr << "Failed to create ImGui fonts texture" << std::endl;
        return false;
    }
    
    return true;
}

void SymbioteGUI::run() {
    // Get swapchain images for rendering
    uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    
    // Create semaphores for synchronization
    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkSemaphore image_available_semaphore;
    VkSemaphore render_finished_semaphore;
    vkCreateSemaphore(device_, &semaphore_info, nullptr, &image_available_semaphore);
    vkCreateSemaphore(device_, &semaphore_info, nullptr, &render_finished_semaphore);
    
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence in_flight_fence;
    vkCreateFence(device_, &fence_info, nullptr, &in_flight_fence);
    
    ImVec4 clear_color = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
    
    while (running_ && !glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        
        // Wait for previous frame
        vkWaitForFences(device_, 1, &in_flight_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &in_flight_fence);
        
        // Acquire next image
        uint32_t image_index;
        VkResult acquire_result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, 
            image_available_semaphore, VK_NULL_HANDLE, &image_index);
        
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Swapchain is out of date (window resized)
            continue;
        } else if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            std::cerr << "Failed to acquire swapchain image" << std::endl;
            break;
        }
        
        // Start ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Process UI
        processUI();
        
        // Handle async generation
        handleAsyncGeneration();
        
        // Rendering
        ImGui::Render();
        
        // Record command buffer
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkBeginCommandBuffer(command_buffers_[image_index], &begin_info);
        
        VkRenderPassBeginInfo render_pass_info = {};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = render_pass_;
        render_pass_info.framebuffer = framebuffers_[image_index];
        render_pass_info.renderArea.extent = {1600, 900}; // TODO: get actual extent
        
        VkClearValue clear_value = {};
        clear_value.color = {{clear_color.x, clear_color.y, clear_color.z, clear_color.w}};
        render_pass_info.clearValueCount = 1;
        render_pass_info.pClearValues = &clear_value;
        
        vkCmdBeginRenderPass(command_buffers_[image_index], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
        
        // Draw ImGui
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffers_[image_index]);
        
        vkCmdEndRenderPass(command_buffers_[image_index]);
        
        if (vkEndCommandBuffer(command_buffers_[image_index]) != VK_SUCCESS) {
            std::cerr << "Failed to record command buffer" << std::endl;
            break;
        }
        
        // Submit command buffer
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &image_available_semaphore;
        submit_info.pWaitDstStageMask = wait_stages;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers_[image_index];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &render_finished_semaphore;
        
        if (vkQueueSubmit(graphics_queue_, 1, &submit_info, in_flight_fence) != VK_SUCCESS) {
            std::cerr << "Failed to submit draw command buffer" << std::endl;
            break;
        }
        
        // Present
        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished_semaphore;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain_;
        present_info.pImageIndices = &image_index;
        
        VkResult present_result = vkQueuePresentKHR(graphics_queue_, &present_info);
        
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
            // Swapchain is out of date
        } else if (present_result != VK_SUCCESS) {
            std::cerr << "Failed to present swapchain image" << std::endl;
            break;
        }
    }
    
    // Cleanup synchronization objects
    vkDeviceWaitIdle(device_);
    vkDestroyFence(device_, in_flight_fence, nullptr);
    vkDestroySemaphore(device_, render_finished_semaphore, nullptr);
    vkDestroySemaphore(device_, image_available_semaphore, nullptr);
}

void SymbioteGUI::processUI() {
    // Menu bar first
    drawMainWindow();
    
    // Draw panels - each in its own window
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
        
        // Status indicator on the right side of menu bar
        ImGui::SameLine(ImGui::GetWindowWidth() - 250);
        if (engine_) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "● MODEL LOADED");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "● NO MODEL");
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
    
    // Header with status - make it very prominent
    ImGui::PushStyleColor(ImGuiCol_Text, engine_ ? IM_COL32(0, 255, 0, 255) : IM_COL32(255, 128, 0, 255));
    if (engine_) {
        ImGui::Text("● MODEL LOADED - Ready to chat!");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Model is loaded and ready for inference");
        }
    } else {
        ImGui::Text("● NO MODEL - Please load a model first (File > Open Model)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Click File menu and select 'Open Model' to load a GGUF file");
        }
    }
    ImGui::PopStyleColor();
    
    ImGui::SameLine();
    ImGui::Text("| Messages: %zu", chat_history_.size());
    
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
    static char input_buffer[4096] = {};
    
    ImGui::PushItemWidth(-ImGui::GetFrameHeightWithSpacing() * 4);
    bool enter_pressed = ImGui::InputText("##Input", input_buffer, sizeof(input_buffer), 
                         ImGuiInputTextFlags_EnterReturnsTrue);
    
    // Check if widget is active to maintain focus
    bool input_active = ImGui::IsItemActive();
    ImGui::PopItemWidth();
    
    ImGui::SameLine();
    bool send_clicked = ImGui::Button("Send", ImVec2(80, 0));
    
    if ((enter_pressed || send_clicked) && strlen(input_buffer) > 0) {
        sendMessage(input_buffer);
        input_buffer[0] = '\0';  // Clear the buffer
        // Keep focus on input after sending
        ImGui::SetKeyboardFocusHere(-1);
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
        bool success = false;
        std::string error_msg;
        
        try {
            loading_status_ = "Creating engine...";
            loading_progress_ = 0.1f;
            
            // Unload current model if any
            if (engine_) {
                loading_status_ = "Unloading previous model...";
                engine_.reset();
            }
            
            loading_status_ = "Creating VulkanSymbioteEngine...";
            loading_progress_ = 0.2f;
            
            // Create new engine with the model path, using GUI's existing Vulkan objects
            // This avoids creating a conflicting Vulkan instance
            try {
                engine_ = std::make_unique<VulkanSymbioteEngine>(
                    path,
                    instance_,
                    physical_device_,
                    device_,
                    compute_queue_,
                    allocator_
                );
                loading_progress_ = 0.5f;
                loading_status_ = "Engine created, loading GGUF...";
                
                // The engine should be ready after construction
                loading_status_ = "Loading model weights...";
                loading_progress_ = 0.7f;
                
                loading_status_ = "Model loaded successfully!";
                loading_progress_ = 1.0f;
                success = true;
                
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
                error_msg = e.what();
                std::cerr << "Failed to create engine: " << error_msg << std::endl;
                success = false;
            }
            
        } catch (const std::exception& e) {
            error_msg = e.what();
            ChatMessage msg;
            msg.type = ChatMessage::ERROR;
            msg.content = std::string("Failed to load model: ") + error_msg;
            msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            std::lock_guard<std::mutex> lock(chat_mutex_);
            chat_history_.push_back(msg);
        }
        
        if (!success && !error_msg.empty()) {
            ChatMessage msg;
            msg.type = ChatMessage::ERROR;
            msg.content = "Failed to load model: " + error_msg;
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
            
            std::string response;
            bool generation_success = false;
            
            // Try to use the engine for actual generation
            if (engine_) {
                try {
                    // Build full prompt from chat history
                    std::string full_prompt = chat_session_->buildPrompt(prompt);
                    
                    // Call the engine to generate
                    response = engine_->generate(full_prompt, 256, 0.7f);
                    generation_success = !response.empty();
                    
                    if (!generation_success) {
                        response = "Error: Generation returned empty response. Please ensure a model is loaded.";
                    }
                } catch (const std::exception& e) {
                    response = "Error during generation: " + std::string(e.what());
                }
            } else {
                // Fallback: engine not ready
                response = "⚠️ No model loaded. Please load a model first using File → Open Model.";
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Add assistant message
            ChatMessage msg;
            msg.type = ChatMessage::ASSISTANT;
            msg.content = response;
            msg.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            msg.token_count = generation_success ? static_cast<uint32_t>(response.size() / 4) : 0;  // Rough estimate
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
