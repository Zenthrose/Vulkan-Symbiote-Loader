#pragma once

#include "Common.h"
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <queue>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <future>
#include <memory>
#include <map>
#include <list>
#include <chrono>

namespace vk_symbiote {

// Forward declarations
class DefragManager;
struct MigrationState;
class TimelineSemaphoreManager;
class MultiQueueManager;
class AsyncOperationManager;

enum class PackType : uint8 {
    ATTENTION_Q = 0, ATTENTION_K = 1, ATTENTION_V = 2, ATTENTION_O = 3,
    FEED_FORWARD_GATE = 4, FEED_FORWARD_UP = 5, FEED_FORWARD_DOWN = 6,
    NORM_GAMMA = 7, NORM_BETA = 8, ROPE = 9, EMBEDDING = 10, HEAD = 11, UNKNOWN = 255
};

enum class MemoryTier : uint8 { VRAM_HOT = 0, RAM_WARM = 1, DISK_COLD = 2, UNLOADED = 3 };

struct PackMetadata {
    uint64 pack_id = 0;
    PackType type = PackType::UNKNOWN;
    uint32 layer_idx = 0, head_idx = 0, head_group_idx = 0;
    size_t compressed_size = 0, decompressed_size = 0;
    uint64 file_offset = 0;
    std::string tensor_name;
    float base_priority = 0.5f;
    uint32 alignment = 256;
    uint32 n_dimensions = 0;
    std::vector<uint64_t> dimensions;
};

struct BufferRegion {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    uint64 offset = 0, size = 0;
    uint32 padding_level = 0;
};

struct MemoryPoolStats {
    uint64 total_size = 0, used_size = 0, free_size = 0;
    uint32 allocation_count = 0;
    float fragmentation_ratio = 0.0f;
};

class FractalMemoryAllocator {
public:
    struct Block {
        uint64 offset = 0, size = 0;
        bool is_free = true;
        uint32 level = 0;
        Block* left = nullptr;
        Block* right = nullptr;
        Block* parent = nullptr;
    };

    FractalMemoryAllocator(VkDevice device, VmaAllocator vma, bool is_vram);
    ~FractalMemoryAllocator();
    Expected<BufferRegion> allocate(uint64 size, uint64 alignment = 256);
    ExpectedVoid deallocate(const BufferRegion& region);
    ExpectedVoid reset();
    MemoryPoolStats get_stats() const;
    uint64 max_allocatable_size() const { return max_block_size_; }
    
    // Time-limited defragmentation for background operation
    void defragment_limited(uint32_t max_time_ms);
    void defragment();
    
    // Complete recursive defragmentation
    void defragment_recursive();
    
private:
    VkDevice device_; VmaAllocator vma_; bool is_vram_;
    uint64_t total_size_ = 0;
    uint64_t max_block_size_ = 0;
    VkBuffer backing_buffer_ = VK_NULL_HANDLE;
    VmaAllocation backing_allocation_ = nullptr;
    Block* root_block_ = nullptr;
    mutable std::mutex mutex_;
    
    Block* split_block_recursive(Block* block, uint64_t size);
    Block* find_best_block(Block* node, uint64_t size, uint64_t);
    Block* find_block_by_offset(Block* node, uint64_t offset);
    Block* find_or_create_block_at(Block* node, uint64_t offset, uint64_t size);
    void merge_adjacent_free_recursive(Block* block);
    void merge_all_free_recursive(Block* node);
    void collect_used_blocks(Block* node, std::vector<std::pair<uint64_t, uint64_t>>& used);
    void destroy_tree(Block* node);
    void calculate_stats_recursive(Block* node, MemoryPoolStats& stats) const;
    uint64_t calculate_usable_size(uint64_t requested, uint64_t alignment);
};

class NomadPack {
public:
    explicit NomadPack(const PackMetadata& metadata, const Path& file_path);
    ~NomadPack();
    const PackMetadata& metadata() const noexcept { return metadata_; }
    ExpectedVoid load_to_ram();
    ExpectedVoid unload_from_ram();
    ExpectedVoid load_to_vram(const BufferRegion& region);
    ExpectedVoid unload_from_vram();
    bool is_in_ram() const noexcept { return ram_data_ != nullptr; }
    bool is_in_vram() const noexcept { return vram_region_.buffer != VK_NULL_HANDLE; }
    float* ram_data() noexcept { return ram_data_; }
    const BufferRegion& vram_region() const noexcept { return vram_region_; }
    BufferRegion& vram_region() noexcept { return vram_region_; }
    size_t ram_footprint() const noexcept;
    size_t vram_footprint() const noexcept;
    size_t compressed_size() const noexcept { return compressed_data_.size(); }
    void set_priority(float priority) { priority_ = priority; }
    float priority() const noexcept { return priority_; }
    void set_tier(MemoryTier tier) { current_tier_ = tier; }
    MemoryTier current_tier() const noexcept { return current_tier_; }
    uint64 last_access_time() const noexcept { return last_access_; }
    void update_access_time() { last_access_ = get_current_time_ns(); }
    uint32 access_count() const noexcept { return access_count_; }
    void increment_access_count() { access_count_++; }
    
    // Async migration support with timeline semaphores
    ExpectedVoid migrate_async(MemoryTier target_tier, std::function<void(bool)> completion_callback = nullptr);
    bool is_migration_complete() const;
    void wait_for_migration();
    
    // Timeline semaphore value for async GPU operations (0 if not using timelines)
    uint64_t get_timeline_value() const;
    
    // Check if currently migrating
    bool is_migrating() const;
    
    // Timeline semaphore enhanced methods
    ExpectedVoid load_to_vram_timeline(VkCommandBuffer transfer_cmd, 
                                        VkDeviceSize size, 
                                        VkBuffer src_buffer,
                                        VkDeviceSize src_offset = 0);
    ExpectedVoid load_to_vram_after_compute(VkCommandBuffer transfer_cmd,
                                             uint64_t wait_compute_timeline,
                                             VkDeviceSize size,
                                             VkBuffer src_buffer);
    void wait_for_migration_timeline();
    bool is_migration_complete_timeline() const;
    
    // Migration implementation
    bool perform_migration(MemoryTier target_tier);
    
    // Vulkan helper methods
    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer cmd_buffer);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    
private:
    PackMetadata metadata_; 
    Path file_path_;
    float* ram_data_ = nullptr; 
    BufferRegion vram_region_;
    std::vector<uint8_t> compressed_data_;
    float priority_ = 0.5f; 
    MemoryTier current_tier_ = MemoryTier::UNLOADED;
    uint64 last_access_ = 0; 
    uint32 access_count_ = 0;
    
    // Vulkan handles
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VmaAllocator vma_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    
    std::unique_ptr<MigrationState> migration_state_;
    
    Expected<std::vector<float>> decompress_zfp();
    Expected<std::vector<float>> decompress_blosc();
    Expected<std::vector<float>> decompress_raw();
    ExpectedVoid read_from_disk(std::vector<uint8_t>& buffer);
    
    // Migration implementations
    bool migrate_to_vram_timeline_impl();
    bool migrate_to_vram_sync(VkBuffer src_buffer, VkDeviceSize size);
};

class DefragManager {
public:
    struct DefragStats {
        uint64_t total_defrags = 0;
        uint64_t total_time_ms = 0;
        float avg_fragmentation_before = 0.0f;
        float avg_fragmentation_after = 0.0f;
        std::chrono::steady_clock::time_point last_defrag;
    };

    DefragManager(FractalMemoryAllocator* vram_alloc, FractalMemoryAllocator* ram_alloc);
    ~DefragManager();
    void start();
    void stop();
    void request_defrag(uint64_t priority = 100);
    void set_interval(uint32_t interval_ms);
    void set_threshold(float fragmentation_threshold);
    bool is_running() const;
    DefragStats get_stats() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

class PackManager {
public:
    PackManager(VkDevice device, VkPhysicalDevice physical_device, VmaAllocator vma);
    ~PackManager();
    ExpectedVoid initialize(uint64 vram_budget, uint64 ram_budget);
    Expected<std::shared_ptr<NomadPack>> get_or_load_pack(uint64 pack_id);
    ExpectedVoid unload_pack(uint64 pack_id);
    ExpectedVoid prefetch_packs(const std::vector<uint64>& pack_ids, float confidence_threshold);
    ExpectedVoid evict_until(uint64 bytes_needed);
    float vram_utilization() const;
    float ram_utilization() const;
    MemoryPoolStats get_vram_stats() const;
    MemoryPoolStats get_ram_stats() const;
    void set_aggression(float aggression) { eviction_aggression_ = std::clamp(aggression, 0.0f, 1.0f); }
    float eviction_aggression() const noexcept { return eviction_aggression_; }
    std::vector<uint64> get_loaded_pack_ids() const;
    size_t total_packs() const noexcept { return packs_.size(); }
    
    // Trigger background defragmentation
    void trigger_defrag();
    
    // Get defrag statistics
    DefragManager::DefragStats get_defrag_stats() const;
    
    // Get timeline semaphore support status
    bool timeline_semaphores_supported() const;
    
    // Timeline semaphore methods
    ExpectedVoid load_pack_with_timeline(uint64_t pack_id, 
                                          VkCommandBuffer transfer_cmd,
                                          VkBuffer src_buffer,
                                          VkDeviceSize size);
    void wait_pack_timeline(uint64_t pack_id);
    void poll_all_timelines();
    
private:
    VkDevice device_; VkPhysicalDevice physical_device_; VmaAllocator vma_;
    std::unordered_map<uint64, std::shared_ptr<NomadPack>> packs_;
    std::unique_ptr<FractalMemoryAllocator> vram_allocator_;
    std::unique_ptr<FractalMemoryAllocator> ram_allocator_;
    std::unique_ptr<DefragManager> defrag_mgr_;
    std::unique_ptr<TimelineSemaphoreManager> timeline_mgr_;
    mutable std::mutex mutex_; float eviction_aggression_ = 0.7f;
    uint64 vram_budget_ = 0, ram_budget_ = 0;
    std::thread prefetch_thread_; std::atomic<bool> prefetch_running_{false};
    std::condition_variable prefetch_cv_; std::queue<uint64> prefetch_queue_;
    void prefetch_worker_loop();
    float calculate_pack_score(const std::shared_ptr<NomadPack>& pack);
};

} // namespace vk_symbiote
