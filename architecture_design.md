# Vulkan Symbiote Engine - Architecture Overview

## Phase 1: High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VulkanSymbioteEngine                         │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  VitalityOracle │◄──►│ PackScheduler   │                    │
│  │ (Relevance+HW)  │    │ (Prefetch/Unload)│                   │
│  └─────────────────┘    └─────────────────┘                    │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              NomadPackManager                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │ WeightPack  │  │ WeightPack  │  │ WeightPack  │ ...   │  │
│  │  │ (Attention)│  │ (MLP Block) │  │ (Norm/Layr) │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  └───────────────────────────────────────────────────────────┘  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ FractalMemory   │    │      VulkanComputePools             │ │
│  │ Allocator       │    │  ┌─────────┐ ┌─────────┐           │ │
│  │ (Suballoc)      │    │  │ Shader  │ │ Buffer  │           │ │
│  │                 │    │  │ Cache   │ │ Pool    │           │ │
│  └─────────────────┘    │  └─────────┘ └─────────┘           │ │
│                         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Backend                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│  │ GGUFLoader  │    │ ZFP/Blosc   │    │ AsyncPrefetcher     │   │
│  │ (Safetens)  │    │ Decompressor│    │ (Background I/O)    │   │
│  └─────────────┘    └─────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 2: Key C++ Classes/Interfaces

### Core Data Structures

#### WeightPack - Atomic unit of model weights
```cpp
struct WeightPack {
    uint64_t pack_id;                    // Unique identifier
    PackType type;                       // ATTENTION_HEAD, MLP_BLOCK, etc.
    uint32_t layer_idx, sub_idx;         // Position in model
    
    // Memory management
    MemoryTier current_tier;             // VRAM_HOT, RAM_WARM, DISK_COLD
    VkBuffer gpu_buffer;                 // GPU memory when loaded
    void* compressed_data;                // Compressed disk data
    void* uncompressed_data;             // Decompressed RAM data
    
    // Metrics for oracle
    uint64_t access_count;               // Usage frequency
    float vitality_score;                // Combined relevance score
    std::chrono::time_point last_access; // Temporal relevance
    
    bool is_resident() const;           // Quick availability check
};
```

#### VitalityOracle - Predictive intelligence
```cpp
class VitalityOracle {
public:
    // Multi-factor scoring
    float calculate_vitality(const WeightPack& pack, 
                           const SemanticContext& context,
                           const HardwareTelemetry& telemetry);
    
    // Machine learning predictor
    void update_predictor(const WeightPack& pack, bool was_useful);
    void predict_next_packs(const SemanticContext& context, 
                          std::vector<uint64_t>& predictions);
    
private:
    // Tiny MLP (4-16-8-1) for prediction
    std::vector<float> mlp_weights_;
    
    // Feature weights (tuned per-device)
    float access_weight_ = 0.4f;      // Historical usage
    float semantic_weight_ = 0.3f;    // Prompt relevance
    float hardware_weight_ = 0.2f;    // Device constraints
    float recency_weight_ = 0.1f;     // Temporal locality
};
```

#### FractalMemoryAllocator - Zero-fragmentation suballocation
```cpp
class FractalMemoryAllocator {
public:
    // Binary tree allocation strategy
    AllocationBlock* allocate(size_t size, size_t alignment);
    void deallocate(VkBuffer buffer, size_t offset);
    
    // Runtime defragmentation
    void defragment();
    float get_fragmentation_ratio() const;
    
private:
    // Hierarchical memory blocks
    struct AllocationBlock {
        VkBuffer buffer;
        size_t size, offset;
        bool is_free;
        std::unique_ptr<AllocationBlock> left_child;
        std::unique_ptr<AllocationBlock> right_child;
    };
    
    std::unique_ptr<AllocationBlock> root_block_;
    size_t total_allocated_, total_size_;
};
```

#### PackScheduler - Intelligent loading/unloading
```cpp
class PackScheduler {
public:
    // Asynchronous operations
    std::future<bool> request_load(uint64_t pack_id);
    void request_unload(uint64_t pack_id, MemoryTier target_tier);
    
    // Background scheduling
    void start_scheduling();
    void stop_scheduling();
    
private:
    // Priority queues based on vitality scores
    std::priority_queue<LoadRequest> load_queue_;
    std::priority_queue<UnloadRequest> unload_queue_;
    
    // Scheduling algorithms
    void process_load_queue();          // Load high-vitality packs
    void process_unload_queue();        // Evict low-vitality packs
    void balance_memory_pressure();     // Maintain VRAM budget
};
```

#### NomadPackManager - Central pack lifecycle
```cpp
class NomadPackManager {
public:
    // Pack access with automatic loading
    std::shared_ptr<WeightPack> get_pack(uint64_t pack_id);
    std::future<std::shared_ptr<WeightPack>> load_pack_async(uint64_t pack_id);
    
    // Intelligent prefetching
    void prefetch_related_packs(uint64_t pack_id, size_t count = 3);
    
    // Memory tier management
    void unload_pack(uint64_t pack_id, MemoryTier target_tier);
    std::unordered_map<MemoryTier, size_t> get_memory_distribution();
    
private:
    // Multi-tier loading strategies
    WeightPack* load_from_disk(uint64_t pack_id);
    WeightPack* load_from_ram(uint64_t pack_id);
    bool evict_to_ram(uint64_t pack_id);
    bool evict_to_disk(uint64_t pack_id);
    
    // Compression with ZFP/Blosc
    std::vector<uint8_t> decompress_pack(const std::vector<uint8_t>& compressed);
    std::vector<uint8_t> compress_pack(const std::vector<uint8_t>& data);
};
```

#### VulkanSymbioteEngine - Main orchestrator
```cpp
class VulkanSymbioteEngine {
public:
    // Core inference interface
    std::vector<float> forward(const std::vector<uint32_t>& input_tokens);
    std::vector<uint32_t> generate(const std::vector<uint32_t>& prompt, 
                                  size_t max_tokens = 100);
    
    // Subsystem access
    FractalMemoryAllocator* get_allocator();
    VitalityOracle* get_oracle();
    PackScheduler* get_scheduler();
    NomadPackManager* get_pack_manager();
    
private:
    // Vulkan setup
    VkDevice device_;
    VmaAllocator vma_allocator_;
    VkQueue compute_queue_;
    
    // Engine subsystems
    std::unique_ptr<FractalMemoryAllocator> allocator_;
    std::unique_ptr<VitalityOracle> oracle_;
    std::unique_ptr<PackScheduler> scheduler_;
    std::unique_ptr<NomadPackManager> pack_manager_;
    
    // Hot-swappable shader system
    std::unordered_map<std::string, VkShaderModule> shader_cache_;
    VkShaderModule compile_shader(const std::string& glsl_source);
    bool reload_shader(const std::string& name, const std::string& glsl_source);
};
```

### Storage Backend

#### GGUFLoader - Model format support
```cpp
class GGUFLoader {
public:
    // Model parsing
    bool load();
    std::vector<TensorInfo> get_tensors() const;
    std::vector<uint8_t> read_tensor_data(const TensorInfo& tensor);
    
    // Intelligent pack generation
    std::vector<std::unique_ptr<WeightPack>> generate_packs(
        std::function<uint64_t(const std::string&)> pack_id_generator);
    
private:
    // GGUF parsing
    bool parse_header();
    bool parse_tensor_info();
    
    // Pack segmentation strategy
    void segment_attention_heads();
    void segment_mlp_blocks();
    void segment_norm_layers();
};
```

### Key Design Principles

1. **Extreme Granularity**: Packs are the smallest unit - typically single attention heads or small MLP blocks (1-50MB each)

2. **Predictive Intelligence**: VitalityOracle uses semantic similarity + hardware telemetry to anticipate needs

3. **Zero Fragmentation**: FractalMemoryAllocator uses binary tree suballocation with runtime defragmentation

4. **Multi-Tier Storage**: VRAM (hot) → RAM (warm) → Compressed disk (cold) → Unloaded (hibernating)

5. **Hot-Swappable Shaders**: Runtime GLSL compilation based on device capabilities

6. **Lossless Compression**: ZFP/Blosc for 2-3x reduction in disk/RAM footprint

This architecture enables running 70B+ models on consumer hardware with only 4-8GB VRAM and 8-32GB RAM while maintaining FP16/FP32 precision.