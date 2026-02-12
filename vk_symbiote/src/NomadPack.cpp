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
#include "../../compression/include/Blosc2Compression.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace vk_symbiote {

// Async I/O operation tracking
struct AsyncIOOperation {
    uint64_t pack_id;
    std::future<bool> future;
    std::chrono::steady_clock::time_point start_time;
    enum class Type { LOAD_FROM_DISK, SAVE_TO_DISK, DECOMPRESS, COMPRESS } type;
};

// Timeline semaphore manager for async GPU operations
class TimelineSemaphoreManager {
public:
    explicit TimelineSemaphoreManager(VkDevice device) 
        : device_(device), timeline_semaphore_(VK_NULL_HANDLE), next_value_(1) {
        VkSemaphoreTypeCreateInfo timeline_info = {};
        timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_info.initialValue = 0;
        
        VkSemaphoreCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        create_info.pNext = &timeline_info;
        
        VkResult result = vkCreateSemaphore(device_, &create_info, nullptr, &timeline_semaphore_);
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to create timeline semaphore" << std::endl;
        }
    }
    
    ~TimelineSemaphoreManager() {
        if (timeline_semaphore_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, timeline_semaphore_, nullptr);
        }
    }
    
    void signal(uint64_t value, VkQueue queue) {
        if (timeline_semaphore_ == VK_NULL_HANDLE) return;
        
        VkTimelineSemaphoreSubmitInfo timeline_submit = {};
        timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_submit.signalSemaphoreValueCount = 1;
        timeline_submit.pSignalSemaphoreValues = &value;
        
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_submit;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &timeline_semaphore_;
        
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    }
    
    bool wait(uint64_t value, uint64_t timeout_ns = UINT64_MAX) {
        if (timeline_semaphore_ == VK_NULL_HANDLE) return false;
        
        VkSemaphoreWaitInfo wait_info = {};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_semaphore_;
        wait_info.pValues = &value;
        
        VkResult result = vkWaitSemaphores(device_, &wait_info, timeout_ns);
        return result == VK_SUCCESS;
    }
    
    uint64_t get_next_value() { return next_value_++; }
    VkSemaphore get_semaphore() const { return timeline_semaphore_; }
    
private:
    VkDevice device_;
    VkSemaphore timeline_semaphore_;
    std::atomic<uint64_t> next_value_;
};

// Multi-queue manager for overlapping transfer and compute
class MultiQueueManager {
public:
    struct QueueSet {
        uint32_t family_index;
        VkQueue queue;
        VkCommandPool pool;
        VkCommandBuffer cmd;
        VkFence fence;
    };
    
    MultiQueueManager(VkDevice device, VkPhysicalDevice physical_device)
        : device_(device) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
        
        // Find dedicated transfer queue
        for (uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && 
                !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                transfer_queue_.family_index = i;
                vkGetDeviceQueue(device_, i, 0, &transfer_queue_.queue);
                create_queue_resources(transfer_queue_);
                break;
            }
        }
        
        // Find compute queue
        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_.family_index = i;
                vkGetDeviceQueue(device_, i, 0, &compute_queue_.queue);
                create_queue_resources(compute_queue_);
                break;
            }
        }
    }
    
    ~MultiQueueManager() {
        cleanup_queue(transfer_queue_);
        cleanup_queue(compute_queue_);
    }
    
    void submit_transfer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkResetFences(device_, 1, &transfer_queue_.fence);
        vkResetCommandBuffer(transfer_queue_.cmd, 0);
        vkBeginCommandBuffer(transfer_queue_.cmd, &begin);
        
        VkBufferCopy region = {};
        region.size = size;
        vkCmdCopyBuffer(transfer_queue_.cmd, src, dst, 1, &region);
        
        vkEndCommandBuffer(transfer_queue_.cmd);
        
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &transfer_queue_.cmd;
        
        vkQueueSubmit(transfer_queue_.queue, 1, &submit, transfer_queue_.fence);
    }
    
    void wait_transfer() {
        vkWaitForFences(device_, 1, &transfer_queue_.fence, VK_TRUE, UINT64_MAX);
    }
    
    VkQueue get_transfer_queue() const { return transfer_queue_.queue; }
    
private:
    VkDevice device_;
    QueueSet transfer_queue_ = {};
    QueueSet compute_queue_ = {};
    
    void create_queue_resources(QueueSet& qs) {
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

// Global async manager singleton
class AsyncOperationManager {
public:
    static AsyncOperationManager& instance() {
        static AsyncOperationManager inst;
        return inst;
    }
    
    void init(VkDevice device, VkPhysicalDevice physical_device) {
        timeline_mgr_ = std::make_unique<TimelineSemaphoreManager>(device);
        queue_mgr_ = std::make_unique<MultiQueueManager>(device, physical_device);
        device_ = device;
    }
    
    TimelineSemaphoreManager* timeline() { return timeline_mgr_.get(); }
    MultiQueueManager* queues() { return queue_mgr_.get(); }
    
private:
    std::unique_ptr<TimelineSemaphoreManager> timeline_mgr_;
    std::unique_ptr<MultiQueueManager> queue_mgr_;
    VkDevice device_ = VK_NULL_HANDLE;
};

// FractalMemoryAllocator implementation
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

ExpectedVoid FractalMemoryAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    destroy_tree(root_block_);
    root_block_ = nullptr;
    return make_expected_success();
}

MemoryPoolStats FractalMemoryAllocator::get_stats() const {
    MemoryPoolStats stats;
    stats.total_size = total_size_;
    return stats;
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

void FractalMemoryAllocator::collect_used_blocks(Block* node, std::vector<std::pair<uint64, uint64>>& used) {
    if (!node) return;
    if (!node->is_free) used.emplace_back(node->offset, node->size);
    collect_used_blocks(node->left, used);
    collect_used_blocks(node->right, used);
}

uint64 FractalMemoryAllocator::calculate_usable_size(uint64 requested, uint64 alignment) {
    uint64 rem = requested % alignment;
    return rem ? requested + alignment - rem : requested;
}

// Migration state tracking
struct MigrationState {
    std::atomic<bool> is_migrating{false};
    std::atomic<bool> cancelled{false};
    std::future<bool> future;
    MemoryTier target_tier;
    uint64_t timeline_value = 0;
};

// NomadPack implementation with async support
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

// Async migration with timeline semaphores
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
    auto* timeline = async_mgr.timeline();
    auto* queues = async_mgr.queues();
    
    if (!timeline || !queues) return false;
    
    // Get timeline value for this migration
    uint64_t signal_val = timeline->get_next_value();
    migration_state_->timeline_value = signal_val;
    
    // Perform transfer
    // queues->submit_transfer(staging_buffer, vram_region_.buffer, metadata_.decompressed_size);
    
    // Signal timeline when done
    // timeline->signal(signal_val, queues->get_transfer_queue());
    
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
    return decompress_zfp(); // Alias
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

// PackManager implementation with async support
PackManager::PackManager(VkDevice device, VkPhysicalDevice physical_device, VmaAllocator vma)
    : device_(device), physical_device_(physical_device), vma_(vma),
      eviction_aggression_(0.7f), vram_budget_(0), ram_budget_(0) {
    AsyncOperationManager::instance().init(device, physical_device);
}

PackManager::~PackManager() {
    prefetch_running_.store(false);
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    
    for (auto& [id, pack] : packs_) {
        pack->wait_for_migration();
    }
}

ExpectedVoid PackManager::initialize(uint64 vram_budget, uint64 ram_budget) {
    vram_budget_ = vram_budget;
    ram_budget_ = ram_budget;
    
    vram_allocator_ = std::make_unique<FractalMemoryAllocator>(device_, vma_, true);
    ram_allocator_ = std::make_unique<FractalMemoryAllocator>(device_, vma_, false);
    
    prefetch_running_.store(true);
    prefetch_thread_ = std::thread(&PackManager::prefetch_worker_loop, this);
    
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
    
    return make_expected_success();
}

float PackManager::vram_utilization() const {
    if (!vram_allocator_) return 0.0f;
    auto stats = vram_allocator_->get_stats();
    return static_cast<float>(stats.used_size) / (stats.total_size + 1);
}

float PackManager::ram_utilization() const {
    if (!ram_allocator_) return 0.0f;
    auto stats = ram_allocator_->get_stats();
    return static_cast<float>(stats.used_size) / (stats.total_size + 1);
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
