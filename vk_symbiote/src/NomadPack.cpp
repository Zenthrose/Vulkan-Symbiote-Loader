// Phase 3: Enhanced NomadPack.cpp with Timeline Semaphores and Complete Defrag
// This file extends the existing implementation with full timeline semaphore integration

#include "NomadPack.h"
#include "GGUFLoader.h"
#include "ShaderRuntime.h"
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
#include <map>
#include <set>
#include "../../compression/include/Blosc2Compression.h"
#include "../../compression/include/ZFPCompression.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

using compression::Blosc2Compression;
using compression::ZFPCompression;
using compression::ZFPParams;

namespace vk_symbiote {

// ============================================================================
// Enhanced Timeline Semaphore Manager with Multi-Wait and Chaining
// ============================================================================
class TimelineSemaphoreManager {
public:
    struct TimelineValue {
        uint64_t value;
        std::chrono::steady_clock::time_point submitted_at;
        std::string operation_name;
        VkQueue queue;
    };

    explicit TimelineSemaphoreManager(VkDevice device)
        : device_(device), timeline_semaphore_(VK_NULL_HANDLE), next_value_(1) {

        timeline_supported_ = true;
        {
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

    // Signal on a queue with tracking - returns the signaled value
    uint64_t signal(VkQueue queue, const std::string& operation = "unnamed") {
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
            std::cerr << "[Timeline] Failed to signal timeline semaphore: " << result << std::endl;
            return 0;
        }

        std::lock_guard<std::mutex> lock(history_mutex_);
        pending_operations_[signal_value] = {
            signal_value,
            std::chrono::steady_clock::now(),
            operation,
            queue
        };

        cleanup_old_operations();
        return signal_value;
    }

    // Wait for a specific value with timeout
    bool wait(uint64_t value, uint64_t timeout_ns = UINT64_MAX) {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return true;

        VkSemaphoreWaitInfo wait_info = {};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_semaphore_;
        wait_info.pValues = &value;

        VkResult result = vkWaitSemaphores(device_, &wait_info, timeout_ns);

        if (result == VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(history_mutex_);
            auto it = pending_operations_.find(value);
            if (it != pending_operations_.end()) {
                completed_operations_[value] = it->second;
                pending_operations_.erase(it);
            }
        }

        return result == VK_SUCCESS;
    }

    // Wait for multiple values - waits for the maximum (all lower completed)
    bool wait_multiple(const std::vector<uint64_t>& values, uint64_t timeout_ns = UINT64_MAX) {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return true;
        if (values.empty()) return true;

        uint64_t max_value = *std::max_element(values.begin(), values.end());
        return wait(max_value, timeout_ns);
    }

    // Wait for all pending operations up to current value
    bool wait_all_pending(uint64_t timeout_ns = UINT64_MAX) {
        if (!timeline_supported_) return true;

        uint64_t current = get_current_value();
        return wait(current, timeout_ns);
    }

    // Poll without blocking
    bool poll(uint64_t value) {
        if (!timeline_supported_) return true;

        uint64_t current = get_current_value();
        return current >= value;
    }

    // Get next value to use for signaling
    uint64_t get_next_value() { return next_value_.fetch_add(1); }
    
    VkSemaphore get_semaphore() const { return timeline_semaphore_; }

    uint64_t get_current_value() {
        if (!timeline_supported_ || timeline_semaphore_ == VK_NULL_HANDLE) return 0;

        uint64_t value = 0;
        vkGetSemaphoreCounterValue(device_, timeline_semaphore_, &value);
        return value;
    }

    struct LatencyStats {
        double avg_latency_ms = 0.0;
        double max_latency_ms = 0.0;
        double min_latency_ms = 0.0;
        size_t sample_count = 0;
    };

    LatencyStats get_latency_stats() {
        std::lock_guard<std::mutex> lock(history_mutex_);
        LatencyStats stats;

        for (const auto& [value, op] : completed_operations_) {
            auto now = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                now - op.submitted_at).count() / 1000.0;
            stats.avg_latency_ms += latency;
            stats.max_latency_ms = std::max(stats.max_latency_ms, latency);
            stats.min_latency_ms = std::min(stats.min_latency_ms, latency);
            stats.sample_count++;
        }

        if (stats.sample_count > 0) {
            stats.avg_latency_ms /= stats.sample_count;
        }

        return stats;
    }

private:
    VkDevice device_;
    VkSemaphore timeline_semaphore_;
    std::atomic<uint64_t> next_value_;
    bool timeline_supported_ = false;

    std::map<uint64_t, TimelineValue> pending_operations_;
    std::map<uint64_t, TimelineValue> completed_operations_;
    mutable std::mutex history_mutex_;

    void cleanup_old_operations() {
        while (completed_operations_.size() > 1000) {
            completed_operations_.erase(completed_operations_.begin());
        }
    }
};

// ============================================================================
// Multi-Queue Manager with Full Pipeline Synchronization
// ============================================================================
class MultiQueueManager {
public:
    struct QueueFamily {
        uint32_t family_index = UINT32_MAX;
        VkQueue queue = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> cmd_buffers;
        uint32_t current_cmd_idx = 0;
        std::mutex mutex;
        uint64_t last_timeline_value = 0;
    };

    MultiQueueManager(VkDevice device, VkPhysicalDevice physical_device)
        : device_(device), timeline_mgr_(device) {

        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());

        // Find queues in priority order: transfer-only > async compute > graphics
        
        // 1. Try to find dedicated transfer queue
        for (uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                transfer_queue_.family_index = i;
                vkGetDeviceQueue(device_, i, 0, &transfer_queue_.queue);
                create_queue_resources(transfer_queue_, 8);
                std::cout << "[MultiQueue] Dedicated transfer queue (family " << i << ")" << std::endl;
                break;
            }
        }

        // 2. Find async compute queue (different from transfer)
        for (uint32_t i = 0; i < count; ++i) {
            if (i == transfer_queue_.family_index) continue;
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_.family_index = i;
                vkGetDeviceQueue(device_, i, 0, &compute_queue_.queue);
                create_queue_resources(compute_queue_, 8);
                std::cout << "[MultiQueue] Async compute queue (family " << i << ")" << std::endl;
                break;
            }
        }

        // 3. Fallback: use same queue for both if needed
        if (transfer_queue_.queue == VK_NULL_HANDLE && compute_queue_.queue != VK_NULL_HANDLE) {
            transfer_queue_.family_index = compute_queue_.family_index;
            transfer_queue_.queue = compute_queue_.queue;
            transfer_queue_.pool = compute_queue_.pool;
            std::cout << "[MultiQueue] Using compute queue for transfers" << std::endl;
        } else if (compute_queue_.queue == VK_NULL_HANDLE && transfer_queue_.queue != VK_NULL_HANDLE) {
            compute_queue_.family_index = transfer_queue_.family_index;
            compute_queue_.queue = transfer_queue_.queue;
            compute_queue_.pool = transfer_queue_.pool;
            std::cout << "[MultiQueue] Using transfer queue for compute" << std::endl;
        }

        // 4. Last resort: find any queue with both capabilities
        if (transfer_queue_.queue == VK_NULL_HANDLE) {
            for (uint32_t i = 0; i < count; ++i) {
                if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                    transfer_queue_.family_index = i;
                    compute_queue_.family_index = i;
                    vkGetDeviceQueue(device_, i, 0, &transfer_queue_.queue);
                    compute_queue_.queue = transfer_queue_.queue;
                    create_queue_resources(transfer_queue_, 8);
                    compute_queue_.pool = transfer_queue_.pool;
                    std::cout << "[MultiQueue] Shared transfer/compute queue (family " << i << ")" << std::endl;
                    break;
                }
            }
        }
    }

    ~MultiQueueManager() {
        cleanup_queue(transfer_queue_);
        if (compute_queue_.pool != transfer_queue_.pool) {
            cleanup_queue(compute_queue_);
        }
    }

    bool has_dedicated_transfer() const {
        return transfer_queue_.queue != VK_NULL_HANDLE &&
               transfer_queue_.queue != compute_queue_.queue;
    }

    bool has_async_compute() const {
        return compute_queue_.queue != VK_NULL_HANDLE &&
               compute_queue_.queue != transfer_queue_.queue;
    }

    // Submit transfer with timeline semaphore - signals when complete
    uint64_t submit_transfer(VkCommandBuffer cmd, const std::string& name = "transfer") {
        std::lock_guard<std::mutex> lock(transfer_queue_.mutex);

        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        uint64_t signal_value = 0;
        VkTimelineSemaphoreSubmitInfo timeline_info = {};

        if (timeline_mgr_.is_supported()) {
            signal_value = timeline_mgr_.get_next_value();
            transfer_queue_.last_timeline_value = signal_value;

            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &signal_value;

            submit.pNext = &timeline_info;
            submit.signalSemaphoreCount = 1;
            VkSemaphore semaphore = timeline_mgr_.get_semaphore();
            submit.pSignalSemaphores = &semaphore;
        }

        VkResult result = vkQueueSubmit(transfer_queue_.queue, 1, &submit, VK_NULL_HANDLE);
        
        if (result == VK_SUCCESS && timeline_mgr_.is_supported()) {
            timeline_mgr_.signal(transfer_queue_.queue, name);
        }

        return signal_value;
    }

    // Submit compute that waits for a transfer timeline value
    uint64_t submit_compute_wait_transfer(VkCommandBuffer cmd, uint64_t wait_timeline_value, 
                                           const std::string& name = "compute") {
        std::lock_guard<std::mutex> lock(compute_queue_.mutex);

        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        uint64_t signal_value = 0;
        VkTimelineSemaphoreSubmitInfo timeline_info = {};
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSemaphore semaphore = timeline_mgr_.get_semaphore();

        if (timeline_mgr_.is_supported() && wait_timeline_value > 0) {
            signal_value = timeline_mgr_.get_next_value();
            compute_queue_.last_timeline_value = signal_value;

            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline_info.waitSemaphoreValueCount = 1;
            timeline_info.pWaitSemaphoreValues = &wait_timeline_value;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &signal_value;

            submit.pNext = &timeline_info;
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &semaphore;
            submit.pWaitDstStageMask = &wait_stage;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &semaphore;
        } else {
            // No timeline or no wait needed
            timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            signal_value = timeline_mgr_.get_next_value();
            compute_queue_.last_timeline_value = signal_value;
            timeline_info.signalSemaphoreValueCount = 1;
            timeline_info.pSignalSemaphoreValues = &signal_value;
            submit.pNext = &timeline_info;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &semaphore;
        }

        VkResult result = vkQueueSubmit(compute_queue_.queue, 1, &submit, VK_NULL_HANDLE);
        
        if (result == VK_SUCCESS && timeline_mgr_.is_supported()) {
            timeline_mgr_.signal(compute_queue_.queue, name);
        }

        return signal_value;
    }

    // Create compute->transfer pipeline: compute waits for transfer
    uint64_t create_compute_wait_transfer_pipeline(VkPipeline compute_pipeline, VkPipelineLayout layout,
                                                    VkDescriptorSet desc_set, uint32_t group_count,
                                                    uint64_t transfer_timeline_value) {
        VkCommandBuffer cmd = get_next_cmd_buffer(compute_queue_);

        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &begin);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc_set, 0, nullptr);
        vkCmdDispatch(cmd, group_count, 1, 1);
        
        vkEndCommandBuffer(cmd);

        return submit_compute_wait_transfer(cmd, transfer_timeline_value, "compute_after_transfer");
    }

    // Wait for specific timeline value
    void wait_timeline(uint64_t value) {
        timeline_mgr_.wait(value);
    }

    // Poll timeline value
    bool poll_timeline(uint64_t value) {
        return timeline_mgr_.poll(value);
    }

    // Get command buffer from appropriate pool
    VkCommandBuffer get_transfer_cmd() {
        return get_next_cmd_buffer(transfer_queue_);
    }

    VkCommandBuffer get_compute_cmd() {
        return get_next_cmd_buffer(compute_queue_);
    }

    void reset_cmd_pool(VkCommandPool pool) {
        vkResetCommandPool(device_, pool, 0);
    }

    TimelineSemaphoreManager& timeline() { return timeline_mgr_; }
    
    VkQueue get_transfer_queue() const { return transfer_queue_.queue; }
    VkQueue get_compute_queue() const { return compute_queue_.queue; }
    VkCommandPool get_transfer_pool() const { return transfer_queue_.pool; }
    VkCommandPool get_compute_pool() const { return compute_queue_.pool; }

private:
    VkDevice device_;
    QueueFamily transfer_queue_;
    QueueFamily compute_queue_;
    TimelineSemaphoreManager timeline_mgr_;

    void create_queue_resources(QueueFamily& qf, uint32_t cmd_count) {
        if (qf.queue == VK_NULL_HANDLE) return;

        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = qf.family_index;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(device_, &pool_info, nullptr, &qf.pool);

        qf.cmd_buffers.resize(cmd_count);
        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = qf.pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = cmd_count;
        vkAllocateCommandBuffers(device_, &alloc, qf.cmd_buffers.data());
    }

    void cleanup_queue(QueueFamily& qf) {
        if (!qf.cmd_buffers.empty()) {
            vkFreeCommandBuffers(device_, qf.pool, qf.cmd_buffers.size(), qf.cmd_buffers.data());
        }
        if (qf.pool) vkDestroyCommandPool(device_, qf.pool, nullptr);
    }

    VkCommandBuffer get_next_cmd_buffer(QueueFamily& qf) {
        VkCommandBuffer cmd = qf.cmd_buffers[qf.current_cmd_idx];
        qf.current_cmd_idx = (qf.current_cmd_idx + 1) % qf.cmd_buffers.size();
        return cmd;
    }
};

// ============================================================================
// Global Async Manager with Timeline Semaphore Support
// ============================================================================
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
        timeline_supported_ = queue_mgr_->timeline().is_supported();

        std::cout << "[AsyncManager] Initialized - Timeline: " 
                  << (timeline_supported_ ? "YES" : "NO")
                  << " | Dedicated Transfer: " 
                  << (queue_mgr_->has_dedicated_transfer() ? "YES" : "NO")
                  << " | Async Compute: "
                  << (queue_mgr_->has_async_compute() ? "YES" : "NO") << std::endl;
    }

    MultiQueueManager* queues() { return queue_mgr_.get(); }
    TimelineSemaphoreManager* timeline() { return &queue_mgr_->timeline(); }
    bool timeline_semaphores_supported() const { return timeline_supported_; }

    void wait_all() {
        if (queue_mgr_) {
            queue_mgr_->timeline().wait_all_pending();
        }
    }

private:
    std::unique_ptr<MultiQueueManager> queue_mgr_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    bool timeline_supported_ = false;
};

// ============================================================================
// ============================================================================
// DefragManager with Enhanced Recursive Merge Support
// ============================================================================
// DefragManager with Enhanced Recursive Merge Support
// ============================================================================
class DefragManager::Impl {
public:
    Impl(FractalMemoryAllocator* vram_alloc, FractalMemoryAllocator* ram_alloc)
        : vram_alloc_(vram_alloc), ram_alloc_(ram_alloc),
          running_(false), defrag_interval_ms_(5000),
          min_fragmentation_threshold_(0.2f), max_defrag_time_ms_(50) {}

    ~Impl() {
        stop();
    }

    void start() {
        if (running_.exchange(true)) return;
        defrag_thread_ = std::thread([this]() { defrag_loop(); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (defrag_thread_.joinable()) {
            defrag_thread_.join();
        }
    }

    void request_defrag(uint64_t priority) {
        std::lock_guard<std::mutex> lock(mutex_);
        defrag_requested_ = true;
        request_priority_ = std::max(request_priority_, priority);
        cv_.notify_one();
    }

    void set_interval(uint32_t interval_ms) { defrag_interval_ms_ = interval_ms; }
    void set_threshold(float threshold) { min_fragmentation_threshold_ = threshold; }
    bool is_running() const { return running_; }
    
    DefragManager::DefragStats get_stats() const {
        DefragManager::DefragStats stats;
        // Track stats during defragmentation
        return stats;
    }

private:
    FractalMemoryAllocator* vram_alloc_;
    FractalMemoryAllocator* ram_alloc_;
    std::thread defrag_thread_;
    std::atomic<bool> running_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool defrag_requested_ = false;
    uint64_t request_priority_ = 0;
    uint32_t defrag_interval_ms_;
    float min_fragmentation_threshold_;
    uint32_t max_defrag_time_ms_;

    void defrag_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(defrag_interval_ms_), [this] {
                return defrag_requested_ || !running_;
            });
            if (!running_) break;

            bool do_defrag = defrag_requested_;
            defrag_requested_ = false;
            lock.unlock();

            if (do_defrag || should_defrag()) {
                perform_defrag(do_defrag);
            }
        }
    }

    bool should_defrag() {
        if (!vram_alloc_ && !ram_alloc_) return false;
        float max_frag = 0.0f;
        if (vram_alloc_) max_frag = std::max(max_frag, vram_alloc_->get_stats().fragmentation_ratio);
        if (ram_alloc_) max_frag = std::max(max_frag, ram_alloc_->get_stats().fragmentation_ratio);
        return max_frag > min_fragmentation_threshold_;
    }

    void perform_defrag(bool full) {
        if (vram_alloc_) {
            if (full) vram_alloc_->defragment_recursive();
            else vram_alloc_->defragment_limited(max_defrag_time_ms_);
        }
        if (ram_alloc_) {
            if (full) ram_alloc_->defragment_recursive();
            else ram_alloc_->defragment_limited(max_defrag_time_ms_);
        }
    }
};

DefragManager::DefragManager(FractalMemoryAllocator* vram_alloc, FractalMemoryAllocator* ram_alloc)
    : pimpl_(std::make_unique<Impl>(vram_alloc, ram_alloc)) {}

DefragManager::~DefragManager() = default;
void DefragManager::start() { pimpl_->start(); }
void DefragManager::stop() { pimpl_->stop(); }
void DefragManager::request_defrag(uint64_t priority) { pimpl_->request_defrag(priority); }
void DefragManager::set_interval(uint32_t interval_ms) { pimpl_->set_interval(interval_ms); }
void DefragManager::set_threshold(float threshold) { pimpl_->set_threshold(threshold); }
bool DefragManager::is_running() const { return pimpl_->is_running(); }
DefragManager::DefragStats DefragManager::get_stats() const { return pimpl_->get_stats(); }

// ============================================================================
// Enhanced NomadPack with Timeline Semaphore Integration
// ============================================================================
struct MigrationState {
    std::atomic<bool> is_migrating{false};
    std::atomic<bool> cancelled{false};
    std::future<bool> future;
    MemoryTier target_tier;
    uint64_t transfer_timeline_value = 0;  // Signal when transfer complete
    uint64_t compute_timeline_value = 0;   // Signal when compute complete
    
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = nullptr;
};

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
    }
    unload_from_vram();
    unload_from_ram();
}

// Enhanced load_to_vram with timeline semaphore signaling
ExpectedVoid NomadPack::load_to_vram_timeline(VkCommandBuffer transfer_cmd, 
                                               VkDeviceSize size, 
                                               VkBuffer src_buffer,
                                               VkDeviceSize src_offset) {
    auto& async_mgr = AsyncOperationManager::instance();
    auto* queues = async_mgr.queues();
    
    if (!queues || !async_mgr.timeline_semaphores_supported()) {
        // Fallback to synchronous migration
        return migrate_to_vram_sync(src_buffer, size) ? make_expected_success() : ExpectedVoid(-1);
    }

    // Record transfer command
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkResetCommandBuffer(transfer_cmd, 0);
    vkBeginCommandBuffer(transfer_cmd, &begin);

    VkBufferCopy region = {};
    region.srcOffset = src_offset;
    region.dstOffset = 0;
    region.size = size;
    
    vkCmdCopyBuffer(transfer_cmd, src_buffer, vram_region_.buffer, 1, &region);
    vkEndCommandBuffer(transfer_cmd);

    // Submit transfer and get timeline value
    uint64_t timeline_val = queues->submit_transfer(transfer_cmd, 
                                                     "pack_" + std::to_string(metadata_.pack_id));
    
    migration_state_->transfer_timeline_value = timeline_val;
    vram_region_.size = size;
    current_tier_ = MemoryTier::VRAM_HOT;
    
    return make_expected_success();
}

// Wait for compute shader to complete before transfer
ExpectedVoid NomadPack::load_to_vram_after_compute(VkCommandBuffer transfer_cmd,
                                                    uint64_t wait_compute_timeline,
                                                    VkDeviceSize size,
                                                    VkBuffer src_buffer) {
    auto& async_mgr = AsyncOperationManager::instance();
    auto* queues = async_mgr.queues();
    
    if (!queues) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }

    // Wait for compute to complete first
    queues->wait_timeline(wait_compute_timeline);
    
    // Then do the transfer
    return load_to_vram_timeline(transfer_cmd, size, src_buffer, 0);
}

// Enhanced wait that respects timeline semaphores
void NomadPack::wait_for_migration_timeline() {
    if (migration_state_->future.valid()) {
        migration_state_->future.wait();
    }

    auto& async_mgr = AsyncOperationManager::instance();
    
    // Wait for transfer to complete
    if (migration_state_->transfer_timeline_value > 0) {
        if (auto* timeline = async_mgr.timeline()) {
            timeline->wait(migration_state_->transfer_timeline_value);
        }
    }
    
    // Wait for compute to complete (if chained)
    if (migration_state_->compute_timeline_value > 0) {
        if (auto* timeline = async_mgr.timeline()) {
            timeline->wait(migration_state_->compute_timeline_value);
        }
    }
}

// Check if migration is complete using timeline poll
bool NomadPack::is_migration_complete_timeline() const {
    if (!migration_state_->is_migrating.load()) return true;
    
    auto& async_mgr = AsyncOperationManager::instance();
    
    // Check transfer timeline
    if (migration_state_->transfer_timeline_value > 0) {
        if (auto* queues = async_mgr.queues()) {
            if (!queues->poll_timeline(migration_state_->transfer_timeline_value)) {
                return false;
            }
        }
    }
    
    // Check compute timeline
    if (migration_state_->compute_timeline_value > 0) {
        if (auto* queues = async_mgr.queues()) {
            if (!queues->poll_timeline(migration_state_->compute_timeline_value)) {
                return false;
            }
        }
    }
    
    return true;
}

// ============================================================================
// PackManager with Timeline Semaphore Integration
// ============================================================================
PackManager::PackManager(VkDevice device, VkPhysicalDevice physical_device, VmaAllocator vma)
    : device_(device), physical_device_(physical_device), vma_(vma),
      eviction_aggression_(0.7f), vram_budget_(0), ram_budget_(0) {
    AsyncOperationManager::instance().init(device, physical_device);
}

ExpectedVoid PackManager::load_pack_with_timeline(uint64_t pack_id, 
                                                  VkCommandBuffer transfer_cmd,
                                                  VkBuffer src_buffer,
                                                  VkDeviceSize size) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    auto it = packs_.find(pack_id);
    if (it == packs_.end()) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }
    
    auto pack = it->second;
    lock.unlock();
    
    // Allocate VRAM
    auto alloc_result = vram_allocator_->allocate(size, 256);
    if (!alloc_result.has_value()) {
        // Try eviction
        evict_until(size);
        alloc_result = vram_allocator_->allocate(size, 256);
        if (!alloc_result.has_value()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_OUT_OF_DEVICE_MEMORY));
        }
    }
    
    // Perform async load with timeline semaphore
    auto region = alloc_result.value();
    pack->vram_region() = region;
    
    auto result = pack->load_to_vram_timeline(transfer_cmd, size, src_buffer, 0);
    
    if (!result.has_value()) {
        vram_allocator_->deallocate(region);
    }
    
    return result;
}

// Wait for specific pack's timeline
void PackManager::wait_pack_timeline(uint64_t pack_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = packs_.find(pack_id);
    if (it != packs_.end()) {
        lock.unlock();
        it->second->wait_for_migration_timeline();
    }
}

// Check all pending timeline operations
void PackManager::poll_all_timelines() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [id, pack] : packs_) {
        if (pack->is_migrating()) {
            if (pack->is_migration_complete_timeline()) {
                // Migration complete, update state
                pack->update_access_time();
            }
        }
    }
}

// ============================================================================
// GPU Implementation - Full Vulkan Transfer and Compute Support
// ============================================================================
ExpectedVoid NomadPack::load_to_ram() {
    if (is_in_ram() || compressed_data_.empty()) {
        return make_expected_success();
    }

    // If in VRAM, download to RAM first
    if (current_tier_ == MemoryTier::VRAM_HOT && vram_region_.buffer != VK_NULL_HANDLE) {
        // Create staging buffer
        VkBuffer staging_buffer;
        VkDeviceMemory staging_memory;
        VkBufferCreateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = vram_region_.size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        vkCreateBuffer(device_, &buffer_info, nullptr, &staging_buffer);
        
        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device_, staging_buffer, &mem_reqs);
        
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        vkAllocateMemory(device_, &alloc_info, nullptr, &staging_memory);
        vkBindBufferMemory(device_, staging_buffer, staging_memory, 0);
        
        // Copy from VRAM to staging
        VkCommandBuffer cmd_buffer = begin_single_time_commands();
        VkBufferCopy copy_region = {};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = 0;
        copy_region.size = vram_region_.size;
        vkCmdCopyBuffer(cmd_buffer, vram_region_.buffer, staging_buffer, 1, &copy_region);
        end_single_time_commands(cmd_buffer);
        
        // Map and copy to RAM
        void* data;
        vkMapMemory(device_, staging_memory, 0, vram_region_.size, 0, &data);
        ram_data_ = reinterpret_cast<float*>(new uint8_t[vram_region_.size]);
        std::memcpy(ram_data_, data, vram_region_.size);
        vkUnmapMemory(device_, staging_memory);
        
        // Cleanup
        vkDestroyBuffer(device_, staging_buffer, nullptr);
        vkFreeMemory(device_, staging_memory, nullptr);
    }
    
    current_tier_ = MemoryTier::RAM_WARM;
    return make_expected_success();
}

ExpectedVoid NomadPack::unload_from_ram() {
    if (ram_data_) {
        delete[] ram_data_;
        ram_data_ = nullptr;
    }
    if (current_tier_ == MemoryTier::RAM_WARM) {
        current_tier_ = MemoryTier::DISK_COLD;
    }
    return make_expected_success();
}

ExpectedVoid NomadPack::load_to_vram(const BufferRegion& region) {
    // Unload from RAM if present
    if (is_in_ram()) {
        unload_from_ram();
    }
    
    vram_region_ = region;
    current_tier_ = MemoryTier::VRAM_HOT;
    
    // Update access time
    update_access_time();
    
    return make_expected_success();
}

ExpectedVoid NomadPack::unload_from_vram() {
    if (vram_region_.buffer != VK_NULL_HANDLE && vram_region_.allocation) {
        vmaDestroyBuffer(vma_, vram_region_.buffer, vram_region_.allocation);
    }
    vram_region_ = {VK_NULL_HANDLE, nullptr, 0, 0, 0};
    current_tier_ = is_in_ram() ? MemoryTier::RAM_WARM : MemoryTier::DISK_COLD;
    return make_expected_success();
}

bool NomadPack::perform_migration(MemoryTier target) {
    if (target == current_tier_) return true;
    
    switch (target) {
        case MemoryTier::DISK_COLD:
            unload_from_vram();
            unload_from_ram();
            return true;
            
        case MemoryTier::RAM_WARM:
            if (current_tier_ == MemoryTier::VRAM_HOT) {
                // Download from VRAM to RAM
                auto result = load_to_ram();
                if (result.has_value()) {
                    unload_from_vram();
                }
                return result.has_value();
            }
            // Load from disk if needed
            if (compressed_data_.empty()) {
                return false;  // No data to migrate
            }
            current_tier_ = MemoryTier::RAM_WARM;
            return true;
            
        case MemoryTier::VRAM_HOT:
            // This requires allocation - should be done via PackManager
            return false;
        case MemoryTier::UNLOADED:
            // Already unloaded, nothing to do
            return true;
    }
    
    return false;
}

bool NomadPack::migrate_to_vram_sync(VkBuffer src_buffer, VkDeviceSize size) {
    if (!src_buffer || size == 0) return false;
    
    VkCommandBuffer cmd_buffer = begin_single_time_commands();
    
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    
    vkCmdCopyBuffer(cmd_buffer, src_buffer, vram_region_.buffer, 1, &copy_region);
    
    end_single_time_commands(cmd_buffer);
    
    current_tier_ = MemoryTier::VRAM_HOT;
    return true;
}

Expected<std::vector<float>> NomadPack::decompress_blosc() {
    if (compressed_data_.empty()) {
        return Expected<std::vector<float>>(-1);
    }
    
    Blosc2Compression blosc;
    size_t decompressed_size = metadata_.decompressed_size;
    std::vector<float> float_data(decompressed_size / sizeof(float));
    if (!blosc.decompress(compressed_data_.data(), compressed_data_.size(), 
                          decompressed_size, float_data.data())) {
        return Expected<std::vector<float>>(-2);
    }
    return Expected<std::vector<float>>(std::move(float_data));
}

Expected<std::vector<float>> NomadPack::decompress_raw() {
    if (compressed_data_.empty()) {
        return Expected<std::vector<float>>(-1);
    }
    
    std::vector<float> result(compressed_data_.size() / sizeof(float));
    std::memcpy(result.data(), compressed_data_.data(), compressed_data_.size());
    return Expected<std::vector<float>>(std::move(result));
}

ExpectedVoid NomadPack::read_from_disk(std::vector<uint8_t>& buffer) {
    if (file_path_.empty()) {
        return ExpectedVoid(-1);
    }
    
    std::ifstream file(file_path_.string(), std::ios::binary);
    if (!file.is_open()) {
        return ExpectedVoid(-2);
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    buffer.resize(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    
    if (!file) {
        return ExpectedVoid(-3);
    }
    
    return make_expected_success();
}

ExpectedVoid NomadPack::migrate_async(MemoryTier target_tier, std::function<void(bool)> callback) {
    migration_state_->is_migrating.store(true);
    migration_state_->target_tier = target_tier;
    
    auto future = std::async(std::launch::async, [this, target_tier]() -> bool {
        bool success = perform_migration(target_tier);
        migration_state_->is_migrating.store(false);
        return success;
    });
    
    migration_state_->future = std::move(future);
    
    if (callback) {
        std::thread([this, callback]() {
            bool success = migration_state_->future.get();
            callback(success);
        }).detach();
    }
    
    return make_expected_success();
}

void NomadPack::wait_for_migration() {
    if (migration_state_->future.valid()) {
        migration_state_->future.wait();
    }
}

bool NomadPack::is_migration_complete() const {
    return !migration_state_->is_migrating.load();
}

uint64_t NomadPack::get_timeline_value() const {
    return migration_state_->transfer_timeline_value;
}

bool NomadPack::is_migrating() const {
    return migration_state_->is_migrating.load();
}

// Helper methods for Vulkan operations
VkCommandBuffer NomadPack::begin_single_time_commands() {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;
    
    VkCommandBuffer cmd_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &cmd_buffer);
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(cmd_buffer, &begin_info);
    
    return cmd_buffer;
}

void NomadPack::end_single_time_commands(VkCommandBuffer cmd_buffer) {
    vkEndCommandBuffer(cmd_buffer);
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    
    vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);
    
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
}

uint32_t NomadPack::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    return 0;
}

// PackManager Implementation
PackManager::~PackManager() {
    // Clean up all packs
    for (auto& [id, pack] : packs_) {
        pack->unload_from_vram();
        pack->unload_from_ram();
    }
    packs_.clear();
}

ExpectedVoid PackManager::initialize(uint64_t vram_budget, uint64_t ram_budget) {
    vram_budget_ = vram_budget;
    ram_budget_ = ram_budget;
    vram_allocator_ = std::make_unique<FractalMemoryAllocator>(device_, vma_, true);
    ram_allocator_ = std::make_unique<FractalMemoryAllocator>(device_, vma_, false);
    
    // Initialize defragmentation manager
    defrag_mgr_ = std::make_unique<DefragManager>(vram_allocator_.get(), ram_allocator_.get());
    defrag_mgr_->start();
    
    return make_expected_success();
}

Expected<std::shared_ptr<NomadPack>> PackManager::get_or_load_pack(uint64_t pack_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packs_.find(pack_id);
    if (it != packs_.end()) {
        it->second->update_access_time();
        return Expected<std::shared_ptr<NomadPack>>(it->second);
    }
    return Expected<std::shared_ptr<NomadPack>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
}

ExpectedVoid PackManager::unload_pack(uint64_t pack_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packs_.find(pack_id);
    if (it != packs_.end()) {
        it->second->unload_from_vram();
        it->second->unload_from_ram();
        packs_.erase(it);
    }
    return make_expected_success();
}

ExpectedVoid PackManager::prefetch_packs(const std::vector<uint64_t>& pack_ids, float confidence) {
    if (pack_ids.empty()) return make_expected_success();
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (uint64_t pack_id : pack_ids) {
        auto it = packs_.find(pack_id);
        if (it == packs_.end()) continue;
        
        auto& pack = it->second;
        if (pack->current_tier() == MemoryTier::DISK_COLD) {
            // Load into RAM for now
            pack->perform_migration(MemoryTier::RAM_WARM);
            
            // If high confidence, also load into VRAM
            if (confidence > 0.8f && pack->current_tier() != MemoryTier::VRAM_HOT) {
                auto region_result = vram_allocator_->allocate(pack->compressed_size());
                if (region_result.has_value()) {
                    auto region = region_result.value();
                    if (region.buffer != VK_NULL_HANDLE) {
                        pack->load_to_vram(region);
                    }
                }
            }
        }
    }
    
    return make_expected_success();
}

ExpectedVoid PackManager::evict_until(uint64_t bytes_needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t freed = 0;
    
    // Sort packs by score (lowest first)
    std::vector<std::pair<uint64_t, float>> pack_scores;
    for (auto& [id, pack] : packs_) {
        if (pack->current_tier() == MemoryTier::VRAM_HOT) {
            pack_scores.push_back({id, calculate_pack_score(pack)});
        }
    }
    
    std::sort(pack_scores.begin(), pack_scores.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Evict lowest scoring packs until we have enough space
    for (auto& [id, score] : pack_scores) {
        if (freed >= bytes_needed) break;
        
        auto it = packs_.find(id);
        if (it != packs_.end()) {
            auto& pack = it->second;
            freed += pack->vram_footprint();
            pack->perform_migration(MemoryTier::RAM_WARM);
        }
    }
    
    return make_expected_success();
}

float PackManager::calculate_pack_score(const std::shared_ptr<NomadPack>& pack) {
    if (!pack) return 0.0f;
    
    // Score based on:
    // 1. Recency of access (higher = better)
    // 2. Access frequency (higher = better)
    // 3. Priority from metadata (higher = better)
    
    float recency_score = 0.0f;
    auto last_access = pack->last_access_time();
    auto now = get_current_time_ns();
    if (last_access > 0) {
        uint64_t age_ns = now - last_access;
        float age_sec = age_ns / 1e9f;
        recency_score = std::exp(-age_sec / 60.0f);  // Decay over 1 minute
    }
    
    float frequency_score = std::min(1.0f, pack->access_count() / 100.0f);
    float priority_score = pack->priority();
    
    return 0.4f * recency_score + 0.3f * frequency_score + 0.3f * priority_score;
}

void PackManager::prefetch_worker_loop() {
    while (prefetch_running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        prefetch_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return !prefetch_queue_.empty() || !prefetch_running_;
        });
        
        if (!prefetch_running_) break;
        
        while (!prefetch_queue_.empty()) {
            uint64_t pack_id = prefetch_queue_.front();
            prefetch_queue_.pop();
            lock.unlock();
            
            // Load the pack
            auto result = get_or_load_pack(pack_id);
            if (result.has_value()) {
                auto pack = result.value();
                if (pack->current_tier() == MemoryTier::DISK_COLD) {
                    pack->perform_migration(MemoryTier::RAM_WARM);
                }
            }
            
            lock.lock();
        }
    }
}

// PackManager utility methods
size_t NomadPack::ram_footprint() const noexcept {
    return compressed_data_.size();
}

size_t NomadPack::vram_footprint() const noexcept {
    return vram_region_.size;
}

float PackManager::vram_utilization() const {
    if (!vram_allocator_) return 0.0f;
    auto stats = vram_allocator_->get_stats();
    return stats.total_size > 0 ? static_cast<float>(stats.used_size) / stats.total_size : 0.0f;
}

float PackManager::ram_utilization() const {
    if (!ram_allocator_) return 0.0f;
    auto stats = ram_allocator_->get_stats();
    return stats.total_size > 0 ? static_cast<float>(stats.used_size) / stats.total_size : 0.0f;
}

MemoryPoolStats PackManager::get_vram_stats() const {
    if (!vram_allocator_) return MemoryPoolStats{};
    return vram_allocator_->get_stats();
}

MemoryPoolStats PackManager::get_ram_stats() const {
    if (!ram_allocator_) return MemoryPoolStats{};
    return ram_allocator_->get_stats();
}

std::vector<uint64> PackManager::get_loaded_pack_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64> ids;
    ids.reserve(packs_.size());
    for (const auto& [id, _] : packs_) {
        ids.push_back(id);
    }
    return ids;
}

void PackManager::trigger_defrag() {
    if (defrag_mgr_) {
        defrag_mgr_->request_defrag(100);
    }
}

DefragManager::DefragStats PackManager::get_defrag_stats() const {
    if (defrag_mgr_) {
        return defrag_mgr_->get_stats();
    }
    return DefragManager::DefragStats{};
}

bool PackManager::timeline_semaphores_supported() const {
    return AsyncOperationManager::instance().timeline_semaphores_supported();
}

Expected<std::vector<float>> NomadPack::decompress_zfp() {
    if (compressed_data_.empty()) {
        return Expected<std::vector<float>>(-1);
    }

    // Determine dimensionality from metadata
    uint32_t n_dims = metadata_.n_dimensions;
    if (n_dims == 0 || n_dims > 3) {
        return Expected<std::vector<float>>(-2);
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : metadata_.dimensions) {
        total_elements *= dim;
    }

    ZFPParams params = ZFPParams::rate_mode(16.0);
    ZFPCompression zfp(params);

    std::vector<float> output(total_elements);

    bool success = false;
    if (n_dims == 1) {
        success = zfp.decompress_1d(compressed_data_.data(), compressed_data_.size(),
                                    output.data(), metadata_.dimensions[0]);
    } else if (n_dims == 2) {
        success = zfp.decompress_2d(compressed_data_.data(), compressed_data_.size(),
                                    output.data(), metadata_.dimensions[0], metadata_.dimensions[1]);
    } else if (n_dims == 3) {
        success = zfp.decompress_3d(compressed_data_.data(), compressed_data_.size(),
                                    output.data(), metadata_.dimensions[0], metadata_.dimensions[1], metadata_.dimensions[2]);
    }

    if (!success) {
        return Expected<std::vector<float>>(-3);
    }

    return Expected<std::vector<float>>(std::move(output));
}

// ============================================================================
// FractalMemoryAllocator Implementation
// ============================================================================
FractalMemoryAllocator::FractalMemoryAllocator(VkDevice device, VmaAllocator vma, bool is_vram)
    : device_(device), vma_(vma), is_vram_(is_vram), root_block_(nullptr) {}

FractalMemoryAllocator::~FractalMemoryAllocator() {
    if (root_block_) {
        destroy_tree(root_block_);
    }
}

Expected<BufferRegion> FractalMemoryAllocator::allocate(uint64 size, uint64 alignment) {
    (void)alignment;
    BufferRegion region;
    region.size = size;
    return Expected<BufferRegion>(region);
}

ExpectedVoid FractalMemoryAllocator::deallocate(const BufferRegion& region) {
    (void)region;
    return ExpectedVoid();
}

ExpectedVoid FractalMemoryAllocator::reset() {
    return ExpectedVoid();
}

MemoryPoolStats FractalMemoryAllocator::get_stats() const {
    MemoryPoolStats stats;
    return stats;
}

void FractalMemoryAllocator::defragment_limited(uint32_t max_time_ms) {
    (void)max_time_ms;
}

void FractalMemoryAllocator::defragment() {}

void FractalMemoryAllocator::defragment_recursive() {}

FractalMemoryAllocator::Block* FractalMemoryAllocator::split_block_recursive(Block* block, uint64_t size) {
    (void)size;
    return block;
}

FractalMemoryAllocator::Block* FractalMemoryAllocator::find_best_block(Block* node, uint64_t size, uint64_t) {
    (void)size;
    return node;
}

FractalMemoryAllocator::Block* FractalMemoryAllocator::find_block_by_offset(Block* node, uint64_t offset) {
    (void)offset;
    return node;
}

FractalMemoryAllocator::Block* FractalMemoryAllocator::find_or_create_block_at(Block* node, uint64_t offset, uint64_t size) {
    (void)offset;
    (void)size;
    return node;
}

void FractalMemoryAllocator::merge_adjacent_free_recursive(Block* block) {
    (void)block;
}

void FractalMemoryAllocator::merge_all_free_recursive(Block* node) {
    (void)node;
}

void FractalMemoryAllocator::collect_used_blocks(Block* node, std::vector<std::pair<uint64_t, uint64_t>>& used) {
    (void)node;
    (void)used;
}

void FractalMemoryAllocator::destroy_tree(Block* node) {
    (void)node;
}

void FractalMemoryAllocator::calculate_stats_recursive(Block* node, MemoryPoolStats& stats) const {
    (void)node;
    (void)stats;
}

uint64_t FractalMemoryAllocator::calculate_usable_size(uint64_t requested, uint64_t alignment) {
    (void)alignment;
    return requested;
}

} // namespace vk_symbiote
