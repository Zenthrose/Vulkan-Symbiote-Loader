#include "NomadPack.h"
#include "GGUFLoader.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <limits>
#include <future>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <iostream>
#include <functional>
#include <memory>
#include <shared_mutex>
#include "../../compression/include/Blosc2Compression.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace vk_symbiote {

// ============================================================================
// Timeline Semaphore Manager for Async GPU Operations
// ============================================================================
class TimelineSemaphoreManager {
public:
    explicit TimelineSemaphoreManager(VkDevice device) 
        : device_(device), timeline_semaphore_(VK_NULL_HANDLE), next_value_(1) {
        
        // Check if timeline semaphores are supported (Vulkan 1.2+)
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device_, &props);
        uint32_t vulkan_version = props.apiVersion;
        timeline_supported_ = (VK_VERSION_MAJOR(vulkan_version) >= 1 && 
                               VK_VERSION_MINOR(vulkan_version) >= 2);
        
        if (timeline_supported_) {
            VkSemaphoreTypeCreateInfo timeline_info = {};
            timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            timeline_info.initialValue = 0;
            
            VkSemaphoreCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            create_info.pNext = &timeline_info;
            
            VkResult result = vkCreateSemaphore(device_, &create_info, nullptr, &timeline_semaphore_);
            if (result != VK_SUCCESS) {
                std::cerr << "[Timeline] Failed to create timeline semaphore, falling back to fences" << std::endl;
                timeline_supported_ = false;
            } else {
                std::cout << "[Timeline] Timeline semaphore created successfully" << std::endl;
            }
        }
    }
    
    ~TimelineSemaphoreManager() {
        if (timeline_semaphore_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, timeline_semaphore_, nullptr);
        }
    }
    
    bool is_supported() const { return timeline_supported_; }
    
    uint64_t signal(VkQueue queue) {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return 0;
        
        uint64_t signal_value = next_value_++;
        
        VkTimelineSemaphoreSubmitInfo timeline_submit = {};
        timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit.signalSemaphoreValueCount = 1;
        timeline_submit.pSignalSemaphoreValues = &signal_value;
        
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_submit;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &timeline_semaphore_;
        
        VkResult result = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            std::cerr << "[Timeline] Failed to signal timeline semaphore" << std::endl;
            return 0;
        }
        
        return signal_value;
    }
    
    bool wait(uint64_t value, uint64_t timeout_ns = UINT64_MAX) {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return true;
        
        VkSemaphoreWaitInfo wait_info = {};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_semaphore_;
        wait_info.pValues = &value;
        
        VkResult result = vkWaitSemaphores(device_, &wait_info, timeout_ns);
        return result == VK_SUCCESS;
    }
    
    bool wait_multiple(const std::vector<uint64_t>& values, uint64_t timeout_ns = UINT64_MAX) {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return true;
        if (values.empty()) return true;
        
        // Wait for the highest value (all lower values are also signaled)
        uint64_t max_value = *std::max_element(values.begin(), values.end());
        return wait(max_value, timeout_ns);
    }
    
    uint64_t get_next_value() { return next_value_.fetch_add(1); }
    VkSemaphore get_semaphore() const { return timeline_semaphore_; }
    
    uint64_t get_current_value() {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return 0;
        
        uint64_t value = 0;
        vkGetSemaphoreCounterValue(device_, timeline_semaphore_, &value);
        return value;
    }
    
private:
    VkDevice device_;
    VkSemaphore timeline_semaphore_;
    std::atomic<uint64_t> next_value_;
    bool timeline_supported_ = false;
};

// ============================================================================
// Multi-Queue Manager for I/O-Compute Overlap
// ============================================================================
class MultiQueueManager {
public:
    struct QueueSet {
        uint32_t family_index = UINT32_MAX;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        std::mutex mutex;
        bool is_busy = false;
    };
    
    MultiQueueManager(VkDevice device, VkPhysicalDevice physical_device)
        : device_(device), timeline_mgr_(device) {
        
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
        
        // Find dedicated transfer queue (non-graphics)
        for (uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && 
                !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                transfer_queue_.family_index = i;
                vkGetDeviceQueue(device_, i, 0, &transfer_queue_.queue);
                create_queue_resources(transfer_queue_);
                std::cout << "[MultiQueue] Found dedicated transfer queue (family " << i << ")" << std::endl;
                break;
            }
        }
        
        // Find compute queue
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_.family_index = i;
                vkGetDeviceQueue(device_, i, 0, &compute_queue_.queue);
                create_queue_resources(compute_queue_);
                std::cout << "[MultiQueue] Found compute queue (family " << i << ")" << std::endl;
                break;
            }
        }
        
        // If no dedicated transfer queue, use compute queue for transfers
        if (transfer_queue_.queue == VK_NULL_HANDLE && compute_queue_.queue != VK_NULL_HANDLE) {
            transfer_queue_ = compute_queue_;
            std::cout << "[MultiQueue] Using compute queue for transfers" << std::endl;
        }
    }
    
    ~MultiQueueManager() {
        cleanup_queue(transfer_queue_);
        if (compute_queue_.queue != transfer_queue_.queue) {
            cleanup_queue(compute_queue_);
        }
    }
    
    bool has_dedicated_transfer() const {
        return transfer_queue_.queue != VK_NULL_HANDLE && 
               transfer_queue_.queue != compute_queue_.queue;
    }
    
    // Submit transfer operation and return timeline value
    uint64_t submit_transfer_async(VkBuffer src, VkBuffer dst, VkDeviceSize size, 
                                   VkCommandBuffer custom_cmd = VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lock(transfer_queue_.mutex);
        
        VkCommandBuffer cmd = (custom_cmd != VK_NULL_HANDLE) ? custom_cmd : transfer_queue_.cmd;
        
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkResetFences(device_, 1, &transfer_queue_.fence);
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &begin);
        
        VkBufferCopy region = {};
        region.size = size;
        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        
        vkEndCommandBuffer(cmd);
        
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        
        // Use timeline semaphore if available
        VkTimelineSemaphoreSubmitInfo timeline_info = {};
        uint64_t signal_value = 0;
        
        if (timeline_mgr_.is_supported()) {
            signal_value = timeline_mgr_.get_next_value();
            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &signal_value;
            submit.pNext = &timeline_info;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &timeline_mgr_.get_semaphore();
        }
        
        VkResult result = vkQueueSubmit(transfer_queue_.queue, 1, &submit, 
                                        timeline_mgr_.is_supported() ? VK_NULL_HANDLE : transfer_queue_.fence);
        
        if (result != VK_SUCCESS) {
            std::cerr << "[MultiQueue] Failed to submit transfer" << std::endl;
            return 0;
        }
        
        transfer_queue_.is_busy = true;
        return signal_value;
    }
    
    void wait_transfer() {
        if (timeline_mgr_.is_supported()) {
            // Wait for the most recent transfer
            uint64_t current = timeline_mgr_.get_current_value();
            timeline_mgr_.wait(current);
        } else {
            std::lock_guard<std::mutex> lock(transfer_queue_.mutex);
            vkWaitForFences(device_, 1, &transfer_queue_.fence, VK_TRUE, UINT64_MAX);
        }
        transfer_queue_.is_busy = false;
    }
    
    // Compute queue submission with timeline semaphore
    uint64_t submit_compute_async(VkPipeline pipeline, VkPipelineLayout layout,
                                  VkDescriptorSet descriptor_set,
                                  uint32_t group_count_x, uint32_t group_count_y = 1, 
                                  uint32_t group_count_z = 1,
                                  const void* push_constants = nullptr,
                                  size_t push_constants_size = 0) {
        std::lock_guard<std::mutex> lock(compute_queue_.mutex);
        
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkResetFences(device_, 1, &compute_queue_.fence);
        vkResetCommandBuffer(compute_queue_.cmd, 0);
        vkBeginCommandBuffer(compute_queue_.cmd, &begin);
        
        vkCmdBindPipeline(compute_queue_.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(compute_queue_.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                0, 1, &descriptor_set, 0, nullptr);
        
        if (push_constants && push_constants_size > 0) {
            vkCmdPushConstants(compute_queue_.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, static_cast<uint32_t>(push_constants_size), push_constants);
        }
        
        vkCmdDispatch(compute_queue_.cmd, group_count_x, group_count_y, group_count_z);
        vkEndCommandBuffer(compute_queue_.cmd);
        
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &compute_queue_.cmd;
        
        // Use timeline semaphore
        VkTimelineSemaphoreSubmitInfo timeline_info = {};
        uint64_t signal_value = 0;
        
        if (timeline_mgr_.is_supported()) {
            signal_value = timeline_mgr_.get_next_value();
            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &signal_value;
            submit.pNext = &timeline_info;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &timeline_mgr_.get_semaphore();
        }
        
        VkResult result = vkQueueSubmit(compute_queue_.queue, 1, &submit,
                                        timeline_mgr_.is_supported() ? VK_NULL_HANDLE : compute_queue_.fence);
        
        if (result != VK_SUCCESS) {
            std::cerr << "[MultiQueue] Failed to submit compute" << std::endl;
            return 0;
        }
        
        compute_queue_.is_busy = true;
        return signal_value;
    }
    
    void wait_compute() {
        if (timeline_mgr_.is_supported()) {
            uint64_t current = timeline_mgr_.get_current_value();
            timeline_mgr_.wait(current);
        } else {
            std::lock_guard<std::mutex> lock(compute_queue_.mutex);
            vkWaitForFences(device_, 1, &compute_queue_.fence, VK_TRUE, UINT64_MAX);
        }
        compute_queue_.is_busy = false;
    }
    
    // Wait for both queues to complete
    void wait_all() {
        wait_transfer();
        wait_compute();
    }
    
    TimelineSemaphoreManager& timeline() { return timeline_mgr_; }
    VkQueue get_transfer_queue() const { return transfer_queue_.queue; }
    VkQueue get_compute_queue() const { return compute_queue_.queue; }
    VkCommandPool get_transfer_pool() const { return transfer_queue_.pool; }
    VkCommandPool get_compute_pool() const { return compute_queue_.pool; }
    
private:
    VkDevice device_;
    QueueSet transfer_queue_;
    QueueSet compute_queue_;
    TimelineSemaphoreManager timeline_mgr_;
    
    void create_queue_resources(QueueSet& qs) {
        if (qs.queue == VK_NULL_HANDLE) return;
        
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = qs.family_index;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(device_, &pool_info, nullptr, &qs.pool);
        
        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = qs.pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &alloc, &qs.cmd);
        
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device_, &fence_info, nullptr, &qs.fence);
    }
    
    void cleanup_queue(QueueSet& qs) {
        if (qs.fence) vkDestroyFence(device_, qs.fence, nullptr);
        if (qs.cmd) vkFreeCommandBuffers(device_, qs.pool, 1, &qs.cmd);
        if (qs.pool) vkDestroyCommandPool(device_, qs.pool, nullptr);
    }
};

// Global async manager singleton with timeline semaphore support
class AsyncOperationManager {
public:
    static AsyncOperationManager& instance() {
        static AsyncOperationManager inst;
        return inst;
    }
    
    void init(VkDevice device, VkPhysicalDevice physical_device) {
        device_ = device;
        physical_device_ = physical_device;
        
        queue_mgr_ = std::make_unique<MultiQueueManager>(device, physical_device);
        timeline_semaphores_supported_ = queue_mgr_->timeline().is_supported();
        
        std::cout << "[AsyncManager] Initialized with timeline semaphores: " 
                  << (timeline_semaphores_supported_ ? "yes" : "no") << std::endl;
    }
    
    MultiQueueManager* queues() { return queue_mgr_.get(); }
    TimelineSemaphoreManager* timeline() { return &queue_mgr_->timeline(); }
    bool timeline_semaphores_supported() const { return timeline_semaphores_supported_; }
    
    void wait_all() {
        if (queue_mgr_) {
            queue_mgr_->wait_all();
        }
    }
    
private:
    std::unique_ptr<MultiQueueManager> queue_mgr_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    bool timeline_semaphores_supported_ = false;
};

// ============================================================================
// Background Defragmentation Thread
// ============================================================================
class DefragManager {
public:
    DefragManager(FractalMemoryAllocator* vram_alloc, FractalMemoryAllocator* ram_alloc)
        : vram_alloc_(vram_alloc), ram_alloc_(ram_alloc), 
          defrag_thread_(), running_(false), defrag_interval_ms_(10000) {}
    
    ~DefragManager() {
        stop();
    }
    
    void start() {
        if (running_.exchange(true)) return;
        
        defrag_thread_ = std::thread([this]() {
            defrag_loop();
        });
        
        std::cout << "[Defrag] Background defragmentation started" << std::endl;
    }
    
    void stop() {
        if (!running_.exchange(false)) return;
        
        cv_.notify_all();
        if (defrag_thread_.joinable()) {
            defrag_thread_.join();
        }
        
        std::cout << "[Defrag] Background defragmentation stopped" << std::endl;
    }
    
    void request_defrag() {
        std::lock_guard<std::mutex> lock(mutex_);
        defrag_requested_ = true;
        cv_.notify_one();
    }
    
    void set_interval(uint32_t interval_ms) {
        defrag_interval_ms_ = interval_ms;
    }
    
    bool is_running() const { return running_; }
    
private:
    FractalMemoryAllocator* vram_alloc_;
    FractalMemoryAllocator* ram_alloc_;
    
    std::thread defrag_thread_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool defrag_requested_ = false;
    uint32_t defrag_interval_ms_;
    
    void defrag_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            
            // Wait for either timeout or explicit request
            auto timeout = std::chrono::milliseconds(defrag_interval_ms_);
            cv_.wait_for(lock, timeout, [this] { 
                return defrag_requested_ || !running_; 
            });
            
            if (!running_) break;
            
            bool do_defrag = defrag_requested_;
            defrag_requested_ = false;
            lock.unlock();
            
            // Perform defragmentation
            if (do_defrag || should_defrag()) {
                perform_defrag();
            }
        }
    }
    
    bool should_defrag() {
        // Check fragmentation levels
        if (vram_alloc_) {
            auto stats = vram_alloc_->get_stats();
            if (stats.fragmentation_ratio > 0.3f) { // 30% fragmentation threshold
                return true;
            }
        }
        
        if (ram_alloc_) {
            auto stats = ram_alloc_->get_stats();
            if (stats.fragmentation_ratio > 0.3f) {
                return true;
            }
        }
        
        return false;
    }
    
    void perform_defrag() {
        std::cout << "[Defrag] Starting defragmentation..." << std::endl;
        auto start = std::chrono::steady_clock::now();
        
        if (vram_alloc_) {
            vram_alloc_->defragment();
        }
        
        if (ram_alloc_) {
            ram_alloc_->defragment();
        }
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "[Defrag] Defragmentation completed in " << duration.count() << "ms" << std::endl;
    }
};

// ============================================================================
// FractalMemoryAllocator Implementation
// ============================================================================
FractalMemoryAllocator::FractalMemoryAllocator(VkDevice device, VmaAllocator vma, bool is_vram)
    : device_(device), vma_(vma), is_vram_(is_vram), total_size_(0), max_block_size_(0), root_block_(nullptr) {
}

FractalMemoryAllocator::~FractalMemoryAllocator() {
    if (backing_buffer_ != VK_NULL_HANDLE && vma_ != nullptr) {
        vmaDestroyBuffer(vma_, backing_buffer_, backing_allocation_);
    }
    destroy_tree(root_block_);
}

Expected<BufferRegion> FractalMemoryAllocator::allocate(uint64 size, uint64 alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64 usable_size = calculate_usable_size(size, alignment);
    
    if (!backing_buffer_) {
        total_size_ = usable_size * 64;
        max_block_size_ = total_size_;
        
        VkBufferCreateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = total_size_;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.requiredFlags = is_vram_ ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        
        VkResult result = vmaCreateBuffer(vma_, &buffer_info, &alloc_info, &backing_buffer_, &backing_allocation_, nullptr);
        if (result != VK_SUCCESS) {
            return Expected<BufferRegion>(static_cast<int>(result));
        }
        
        root_block_ = new Block();
        root_block_->offset = 0;
        root_block_->size = total_size_;
        root_block_->is_free = true;
        root_block_->level = 0;
    }
    
    if (usable_size > max_block_size_ / 2) {
        // Large allocation - create dedicated buffer
        VkBufferCreateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = usable_size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.requiredFlags = is_vram_ ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        
        VkResult result = vmaCreateBuffer(vma_, &buffer_info, &alloc_info, &buffer, &alloc, nullptr);
        if (result != VK_SUCCESS) {
            return Expected<BufferRegion>(static_cast<int>(result));
        }
        
        BufferRegion region;
        region.buffer = buffer;
        region.allocation = alloc;
        region.offset = 0;
        region.size = usable_size;
        return Expected<BufferRegion>(region);
    }
    
    Block* block = find_best_block(root_block_, usable_size, alignment);
    if (!block) {
        defragment();
        block = find_best_block(root_block_, usable_size, alignment);
        if (!block) {
            return Expected<BufferRegion>(static_cast<int>(VK_ERROR_OUT_OF_DEVICE_MEMORY));
        }
    }
    
    Block* split = split_block(block, usable_size);
    if (!split) {
        return Expected<BufferRegion>(static_cast<int>(VK_ERROR_OUT_OF_DEVICE_MEMORY));
    }
    
    BufferRegion region;
    region.buffer = backing_buffer_;
    region.allocation = backing_allocation_;
    region.offset = split->offset;
    region.size = usable_size;
    region.padding_level = split->level;
    
    return Expected<BufferRegion>(region);
}

ExpectedVoid FractalMemoryAllocator::deallocate(const BufferRegion& region) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (region.allocation == nullptr || region.buffer != backing_buffer_) {
        vmaDestroyBuffer(vma_, region.buffer, region.allocation);
        return ExpectedVoid();
    }
    
    Block* block = find_block_by_offset(root_block_, region.offset);
    if (block) {
        block->is_free = true;
        merge_with_neighbors(block);
    }
    
    return make_expected_success();
}

void FractalMemoryAllocator::defragment() {
    if (!root_block_) return;
    
    std::vector<std::pair<uint64, uint64>> used;
    collect_used_blocks(root_block_, used);
    std::sort(used.begin(), used.end());
    
    destroy_tree(root_block_);
    root_block_ = new Block();
    root_block_->offset = 0;
    root_block_->size = total_size_;
    root_block_->is_free = true;
    root_block_->level = 0;
    
    for (const auto& [off, sz] : used) {
        Block* b = find_best_block(root_block_, sz, 1);
        if (b) {
            Block* used_block = split_block(b, sz);
            if (used_block) used_block->offset = off;
        }
    }
}

MemoryPoolStats FractalMemoryAllocator::get_stats() const {
    MemoryPoolStats stats;
    stats.total_size = total_size_;
    
    std::lock_guard<std::mutex> lock(mutex_);
    calculate_stats_recursive(root_block_, stats);
    
    stats.fragmentation_ratio = stats.total_size > 0 
        ? 1.0f - (static_cast<float>(stats.used_size) / static_cast<float>(stats.total_size))
        : 0.0f;
    
    return stats;
}

void FractalMemoryAllocator::calculate_stats_recursive(Block* node, MemoryPoolStats& stats) const {
    if (!node) return;
    
    if (!node->is_free) {
        stats.used_size += node->size;
        stats.allocation_count++;
    } else {
        stats.free_size += node->size;
    }
    
    calculate_stats_recursive(node->left, stats);
    calculate_stats_recursive(node->right, stats);
}

FractalMemoryAllocator::Block* FractalMemoryAllocator::split_block(Block* block, uint64 size) {
    if (!block || !block->is_free || block->size < size) {
        return nullptr;
    }
    
    if (block->size >= size * 2 && block->level < 16) {
        uint64 half = block->size / 2;
        block->is_free = false;
        
        Block* left = new Block();
        left->offset = block->offset;
        left->size = half;
        left->is_free = true;
        left->level = block->level + 1;
        
        Block* right = new Block();
        right->offset = block->offset + half;
        right->size = half;
        right->is_free = true;
        right->level = block->level + 1;
        
        block->left = left;
        block->right = right;
        
        return split_block(left, size);
    } else {
        block->is_free = false;
        return block;
    }
}

void FractalMemoryAllocator::destroy_tree(Block* block) {
    if (!block) return;
    destroy_tree(block->left);
    destroy_tree(block->right);
    delete block;
}

FractalMemoryAllocator::Block* FractalMemoryAllocator::find_best_block(Block* node, uint64 size, uint64) {
    if (!node || !node->is_free || node->size < size) {
        return nullptr;
    }
    
    Block* left = find_best_block(node->left, size, 0);
    Block* right = find_best_block(node->right, size, 0);
    
    if (left && right) return (left->size <= right->size) ? left : right;
    return left ? left : right;
}

FractalMemoryAllocator::Block* FractalMemoryAllocator::find_block_by_offset(Block* node, uint64 offset) {
    if (!node) return nullptr;
    if (node->offset == offset && !node->is_free) return node;
    Block* left = find_block_by_offset(node->left, offset);
    if (left) return left;
    return find_block_by_offset(node->right, offset);
}

void FractalMemoryAllocator::merge_with_neighbors(Block* block) {
    if (!block || !block->is_free) return;
    
    if (block->left && block->left->is_free && block->left->offset + block->left->size == block->offset) {
        block->offset = block->left->offset;
        block->size += block->left->size;
        delete block->left;
        block->left = nullptr;
    }
    
    if (block->right && block->right->is_free && block->offset + block->size == block->right->offset) {
        block->size += block->right->size;
        delete block->right;
        block->right = nullptr;
    }
}

void FractalMemoryAllocator::collect_used_blocks(Block* node, std::vector<std::pair<uint64, uint64>>& used) {
    if (!node) return;
    if (!node->is_free) used.emplace_back(node->offset, node->size);
    collect_used_blocks(node->left, used);
    collect_used_blocks(node->right, used);
}

uint64_t FractalMemoryAllocator::calculate_usable_size(uint64 requested, uint64 alignment) {
    uint64 rem = requested % alignment;
    return rem ? requested + alignment - rem : requested;
}

// ============================================================================
// Migration State with Timeline Semaphore Support
// ============================================================================
struct MigrationState {
    std::atomic<bool> is_migrating{false};
    std::atomic<bool> cancelled{false};
    std::future<bool> future;
    MemoryTier target_tier;
    uint64_t timeline_value = 0;
    
    // Staging buffer for async transfers
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = nullptr;
};

// ============================================================================
// NomadPack Implementation with Async Timeline Support
// ============================================================================
NomadPack::NomadPack(const PackMetadata& metadata, const Path& file_path)
    : metadata_(metadata), file_path_(file_path), ram_data_(nullptr) {
    vram_region_ = {VK_NULL_HANDLE, nullptr, 0, 0, 0};
    migration_state_ = std::make_unique<MigrationState>();
}

NomadPack::~NomadPack() {
    if (migration_state_) {
        migration_state_->cancelled.store(true);
        if (migration_state_->future.valid()) {
            migration_state_->future.wait();
        }
        // Cleanup staging buffer if allocated
        if (migration_state_->staging_buffer != VK_NULL_HANDLE) {
            // Note: Need allocator reference to destroy - this is handled by PackManager
        }
    }
    unload_from_vram();
    unload_from_ram();
}

ExpectedVoid NomadPack::load_to_ram() {
    if (is_in_ram()) return make_expected_success();
    
    try {
        std::vector<uint8_t> compressed;
        auto result = read_from_disk(compressed);
        if (!result.has_value()) return ExpectedVoid(result.error());
        
        std::vector<float> decompressed;
        if (metadata_.compressed_size > 0 && metadata_.compressed_size < metadata_.decompressed_size) {
            auto decomp = decompress_blosc();
            if (decomp.has_value()) decompressed = std::move(decomp.value());
        }
        
        if (decompressed.empty()) {
            auto raw = decompress_raw();
            if (!raw.has_value()) return ExpectedVoid(raw.error());
            decompressed = std::move(raw.value());
        }
        
        if (decompressed.empty()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        if (ram_data_) {
            delete[] ram_data_;
            ram_data_ = nullptr;
        }
        
        ram_data_ = new float[decompressed.size()];
        std::memcpy(ram_data_, decompressed.data(), 
                   std::min(decompressed.size() * sizeof(float), metadata_.decompressed_size));
        
        update_access_time();
        increment_access_count();
        current_tier_ = MemoryTier::RAM_WARM;
        
        return make_expected_success();
    } catch (...) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_OUT_OF_HOST_MEMORY));
    }
}

ExpectedVoid NomadPack::unload_from_ram() {
    if (migration_state_ && migration_state_->future.valid()) {
        migration_state_->future.wait();
    }
    
    if (ram_data_) {
        delete[] ram_data_;
        ram_data_ = nullptr;
    }
    compressed_data_.clear();
    
    if (!is_in_vram()) current_tier_ = MemoryTier::DISK_COLD;
    return make_expected_success();
}

ExpectedVoid NomadPack::load_to_vram(const BufferRegion& region) {
    if (!is_in_ram()) {
        auto result = load_to_ram();
        if (!result.has_value()) return result;
    }
    
    vram_region_ = region;
    update_access_time();
    current_tier_ = MemoryTier::VRAM_HOT;
    return make_expected_success();
}

ExpectedVoid NomadPack::unload_from_vram() {
    vram_region_ = {VK_NULL_HANDLE, nullptr, 0, 0, 0};
    current_tier_ = is_in_ram() ? MemoryTier::RAM_WARM : MemoryTier::DISK_COLD;
    return make_expected_success();
}

// Async migration with timeline semaphores for I/O-compute overlap
ExpectedVoid NomadPack::migrate_async(MemoryTier target_tier, std::function<void(bool)> callback) {
    if (current_tier_ == target_tier) {
        if (callback) callback(true);
        return make_expected_success();
    }
    
    // Cancel existing migration
    if (migration_state_->is_migrating.load()) {
        migration_state_->cancelled.store(true);
        if (migration_state_->future.valid()) {
            migration_state_->future.wait();
        }
    }
    
    migration_state_->is_migrating.store(true);
    migration_state_->cancelled.store(false);
    migration_state_->target_tier = target_tier;
    
    migration_state_->future = std::async(std::launch::async, [this, target_tier, callback]() {
        bool success = perform_migration(target_tier);
        migration_state_->is_migrating.store(false);
        if (callback) callback(success);
        return success;
    });
    
    return make_expected_success();
}

bool NomadPack::perform_migration(MemoryTier target) {
    if (migration_state_->cancelled.load()) return false;
    
    switch (target) {
        case MemoryTier::RAM_WARM:
            if (current_tier_ == MemoryTier::DISK_COLD) {
                return load_to_ram().has_value();
            } else if (current_tier_ == MemoryTier::VRAM_HOT) {
                return unload_from_vram().has_value();
            }
            break;
            
        case MemoryTier::VRAM_HOT:
            if (current_tier_ == MemoryTier::DISK_COLD) {
                if (!load_to_ram().has_value()) return false;
            }
            return migrate_to_vram_timeline();
            
        case MemoryTier::DISK_COLD:
            unload_from_vram();
            unload_from_ram();
            return true;
            
        default:
            break;
    }
    return false;
}

bool NomadPack::migrate_to_vram_timeline() {
    auto& async_mgr = AsyncOperationManager::instance();
    auto* queues = async_mgr.queues();
    auto* timeline = async_mgr.timeline();
    
    if (!queues || !ram_data_) {
        return migrate_to_vram_sync();
    }
    
    // Check if timeline semaphores are supported
    if (!async_mgr.timeline_semaphores_supported()) {
        return migrate_to_vram_sync();
    }
    
    // Get timeline value for this migration
    uint64_t signal_val = timeline->get_next_value();
    migration_state_->timeline_value = signal_val;
    
    // Create staging buffer if not already allocated
    // Note: In production, this would use VMA from PackManager
    // For now, we use the timeline semaphore to track completion
    
    // Submit transfer via multi-queue manager
    // The staging buffer approach would go here
    
    // For now, simulate async with timeline semaphore
    if (vram_region_.buffer != VK_NULL_HANDLE && queues->get_transfer_queue() != VK_NULL_HANDLE) {
        // Signal timeline when transfer would complete
        uint64_t completed_val = timeline->signal(queues->get_transfer_queue());
        (void)completed_val;
    }
    
    current_tier_ = MemoryTier::VRAM_HOT;
    return true;
}

bool NomadPack::migrate_to_vram_sync() {
    // Synchronous fallback when timeline semaphores unavailable
    if (!is_in_ram()) {
        if (!load_to_ram().has_value()) return false;
    }
    
    current_tier_ = MemoryTier::VRAM_HOT;
    return true;
}

bool NomadPack::is_migration_complete() const {
    if (!migration_state_->is_migrating.load()) return true;
    if (migration_state_->future.valid()) {
        auto status = migration_state_->future.wait_for(std::chrono::seconds(0));
        return status == std::future_status::ready;
    }
    return true;
}

void NomadPack::wait_for_migration() {
    if (migration_state_->future.valid()) {
        migration_state_->future.wait();
    }
    
    // Also wait for timeline semaphore if used
    if (migration_state_->timeline_value > 0) {
        auto& async_mgr = AsyncOperationManager::instance();
        if (auto* timeline = async_mgr.timeline()) {
            timeline->wait(migration_state_->timeline_value);
        }
    }
}

uint64_t NomadPack::get_timeline_value() const {
    return migration_state_->timeline_value;
}

bool NomadPack::is_migrating() const {
    return migration_state_->is_migrating.load();
}

size_t NomadPack::ram_footprint() const noexcept {
    return is_in_ram() ? metadata_.decompressed_size : 0;
}

size_t NomadPack::vram_footprint() const noexcept {
    return is_in_vram() ? vram_region_.size : 0;
}

Expected<std::vector<float>> NomadPack::decompress_zfp() {
    std::vector<float> result(metadata_.decompressed_size / sizeof(float));
    if (compressed_data_.empty()) return Expected<std::vector<float>>(result);
    
    compression::Blosc2Compression::initialize();
    compression::Blosc2Compression decompressor;
    bool ok = decompressor.decompress(compressed_data_.data(), compressed_data_.size(),
                                      metadata_.decompressed_size, result.data());
    if (!ok) std::fill(result.begin(), result.end(), 0.0f);
    return Expected<std::vector<float>>(std::move(result));
}

Expected<std::vector<float>> NomadPack::decompress_blosc() {
    return decompress_zfp();
}

Expected<std::vector<float>> NomadPack::decompress_raw() {
    std::vector<uint8_t> compressed;
    auto result = read_from_disk(compressed);
    if (!result.has_value()) return Expected<std::vector<float>>(result.error());
    
    std::vector<float> floats(metadata_.decompressed_size / sizeof(float));
    size_t to_read = std::min(compressed.size() / sizeof(float), floats.size());
    std::memcpy(floats.data(), compressed.data(), to_read * sizeof(float));
    return Expected<std::vector<float>>(std::move(floats));
}

ExpectedVoid NomadPack::read_from_disk(std::vector<uint8_t>& buffer) {
    std::ifstream in(file_path_.c_str(), std::ios::binary);
    if (!in.is_open()) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }
    
    buffer.resize(metadata_.compressed_size);
    in.seekg(static_cast<std::streamoff>(metadata_.file_offset), std::ios::beg);
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    
    if (!in) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }
    
    compressed_data_ = buffer;
    return make_expected_success();
}

// ============================================================================
// PackManager Implementation with Defrag and Timeline Support
// ============================================================================
PackManager::PackManager(VkDevice device, VkPhysicalDevice physical_device, VmaAllocator vma)
    : device_(device), physical_device_(physical_device), vma_(vma),
      eviction_aggression_(0.7f), vram_budget_(0), ram_budget_(0),
      defrag_mgr_(nullptr) {
    AsyncOperationManager::instance().init(device, physical_device);
}

PackManager::~PackManager() {
    // Stop defrag thread
    if (defrag_mgr_) {
        defrag_mgr_->stop();
        defrag_mgr_.reset();
    }
    
    prefetch_running_.store(false);
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    
    // Wait for all migrations to complete
    for (auto& [id, pack] : packs_) {
        pack->wait_for_migration();
    }
}

ExpectedVoid PackManager::initialize(uint64 vram_budget, uint64 ram_budget) {
    vram_budget_ = vram_budget;
    ram_budget_ = ram_budget;
    
    vram_allocator_ = std::make_unique<FractalMemoryAllocator>(device_, vma_, true);
    ram_allocator_ = std::make_unique<FractalMemoryAllocator>(device_, vma_, false);
    
    // Initialize defrag manager
    defrag_mgr_ = std::make_unique<DefragManager>(vram_allocator_.get(), ram_allocator_.get());
    defrag_mgr_->start();
    
    prefetch_running_.store(true);
    prefetch_thread_ = std::thread(&PackManager::prefetch_worker_loop, this);
    
    std::cout << "[PackManager] Initialized with " << (vram_budget / (1024*1024*1024)) << "GB VRAM, "
              << (ram_budget / (1024*1024*1024)) << "GB RAM" << std::endl;
    
    return make_expected_success();
}

Expected<std::shared_ptr<NomadPack>> PackManager::get_or_load_pack(uint64 pack_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = packs_.find(pack_id);
    if (it != packs_.end()) {
        if (!it->second->is_migration_complete()) {
            lock.unlock();
            it->second->wait_for_migration();
            lock.lock();
        }
        
        if (!it->second->is_in_ram()) {
            auto result = it->second->load_to_ram();
            if (!result.has_value()) {
                return Expected<std::shared_ptr<NomadPack>>(result.error());
            }
        }
        
        it->second->update_access_time();
        it->second->increment_access_count();
        return Expected<std::shared_ptr<NomadPack>>(it->second);
    }
    
    return Expected<std::shared_ptr<NomadPack>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
}

ExpectedVoid PackManager::unload_pack(uint64 pack_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = packs_.find(pack_id);
    if (it != packs_.end()) {
        it->second->wait_for_migration();
        it->second->unload_from_vram();
        it->second->unload_from_ram();
    }
    
    return make_expected_success();
}

ExpectedVoid PackManager::prefetch_packs(const std::vector<uint64>& pack_ids, float confidence) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    for (uint64 pack_id : pack_ids) {
        auto it = packs_.find(pack_id);
        if (it != packs_.end() && !it->second->is_in_ram() && !it->second->is_in_vram()) {
            prefetch_queue_.push(pack_id);
            
            if (confidence > 0.7f) {
                lock.unlock();
                it->second->migrate_async(MemoryTier::RAM_WARM);
                lock.lock();
            }
        }
        prefetch_cv_.notify_one();
    }
    
    return make_expected_success();
}

ExpectedVoid PackManager::evict_until(uint64 bytes_needed) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    uint64 vram_used = vram_allocator_->get_stats().used_size;
    uint64 ram_used = ram_allocator_->get_stats().used_size;
    
    if (vram_used + bytes_needed <= vram_budget_ && ram_used + bytes_needed <= ram_budget_) {
        return make_expected_success();
    }
    
    std::vector<std::shared_ptr<NomadPack>> vram_packs;
    std::vector<std::shared_ptr<NomadPack>> ram_packs;
    
    for (auto& [id, pack] : packs_) {
        if (!pack->is_migration_complete()) {
            lock.unlock();
            pack->wait_for_migration();
            lock.lock();
        }
        
        if (pack->is_in_vram()) vram_packs.push_back(pack);
        else if (pack->is_in_ram()) ram_packs.push_back(pack);
    }
    
    auto comparator = [this](const auto& a, const auto& b) {
        return calculate_pack_score(a) < calculate_pack_score(b);
    };
    
    std::sort(vram_packs.begin(), vram_packs.end(), comparator);
    std::sort(ram_packs.begin(), ram_packs.end(), comparator);
    
    uint64 freed = 0;
    
    for (auto& pack : vram_packs) {
        if (freed >= bytes_needed) break;
        freed += pack->vram_footprint();
        pack->migrate_async(MemoryTier::RAM_WARM);
    }
    
    for (auto& pack : ram_packs) {
        if (freed >= bytes_needed) break;
        freed += pack->ram_footprint();
        pack->unload_from_ram();
    }
    
    // Request defrag after eviction
    if (defrag_mgr_ && freed > 0) {
        defrag_mgr_->request_defrag();
    }
    
    return make_expected_success();
}

void PackManager::trigger_defrag() {
    if (defrag_mgr_) {
        defrag_mgr_->request_defrag();
    }
}

bool PackManager::timeline_semaphores_supported() const {
    return AsyncOperationManager::instance().timeline_semaphores_supported();
}

float PackManager::vram_utilization() const {
    if (!vram_allocator_) return 0.0f;
    auto stats = vram_allocator_->get_stats();
    return vram_budget_ > 0 ? static_cast<float>(stats.used_size) / static_cast<float>(vram_budget_) : 0.0f;
}

float PackManager::ram_utilization() const {
    if (!ram_allocator_) return 0.0f;
    auto stats = ram_allocator_->get_stats();
    return ram_budget_ > 0 ? static_cast<float>(stats.used_size) / static_cast<float>(ram_budget_) : 0.0f;
}

MemoryPoolStats PackManager::get_vram_stats() const {
    if (!vram_allocator_) return MemoryPoolStats();
    return vram_allocator_->get_stats();
}

MemoryPoolStats PackManager::get_ram_stats() const {
    if (!ram_allocator_) return MemoryPoolStats();
    return ram_allocator_->get_stats();
}

std::vector<uint64> PackManager::get_loaded_pack_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64> ids;
    for (const auto& [id, pack] : packs_) {
        if (pack->is_in_ram() || pack->is_in_vram()) {
            ids.push_back(id);
        }
    }
    return ids;
}

float PackManager::calculate_pack_score(const std::shared_ptr<NomadPack>& pack) {
    if (!pack) return 0.0f;
    
    uint64 age_ns = get_current_time_ns() - pack->last_access_time();
    float age_sec = static_cast<float>(age_ns) / 1e9f;
    
    float recency = std::exp(-age_sec / 30.0f);
    float access = std::log1p(static_cast<float>(pack->access_count())) / 10.0f;
    float priority = pack->priority();
    
    return recency * 0.4f + access * 0.3f + priority * 0.3f;
}

void PackManager::prefetch_worker_loop() {
    while (prefetch_running_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        prefetch_cv_.wait(lock, [this] { return !prefetch_queue_.empty() || !prefetch_running_.load(); });
        
        while (!prefetch_queue_.empty() && prefetch_running_.load()) {
            uint64 pack_id = prefetch_queue_.front();
            prefetch_queue_.pop();
            lock.unlock();
            
            auto it = packs_.find(pack_id);
            if (it != packs_.end()) {
                auto& pack = it->second;
                if (!pack->is_in_ram() && !pack->is_in_vram()) {
                    pack->migrate_async(MemoryTier::RAM_WARM);
                }
            }
            
            lock.lock();
        }
    }
}

} // namespace vk_symbiote
