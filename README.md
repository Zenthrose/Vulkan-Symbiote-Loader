# Vulkan Symbiote Engine

An extreme memory-efficient LLM inference engine designed to run large unquantized models on consumer hardware with minimal VRAM (4-8 GB) and RAM (8-32 GB).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Developer Guide](#developer-guide)

## Architecture Overview

### Core Components

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
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │ │
│  │ FractalMemory   │    │      VulkanComputePools             │ │
│  │ Allocator       │    │  ┌─────────┐ ┌─────────┐           │ │
│  │ (Suballoc)      │    │  │ Shader  │ │ Buffer  │           │ │
│  │                 │    │  │ Cache   │ │ Pool    │           │ │
│  └─────────────────┘    │  └─────────┘ └─────────┘           │ │
│                         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│                    Storage Backend                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│  │ GGUFLoader  │    │ ZFP/Blosc   │    │ AsyncPrefetcher     │   │
│  │ (Safetens) │    │ Decompressor│    │ (Background I/O)    │   │
│  └─────────────┘    └─────────────┘    └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Pack-based Memory Management**: Loads weights in small, granular "packs" (1-50MB each)
- **Extreme Memory Efficiency**: Never loads entire model into memory
- **Pure Vulkan Compute**: No CUDA/ROCm dependencies
- **Predictive Prefetching**: Smart anticipation of needed weight packs
- **Fractal Memory Allocation**: Zero-fragmentation suballocation
- **Hot-swappable Shaders**: Runtime SPIR-V compilation
- **Lossless Compression**: ZFP/Blosc2 for 2-3x reduction
- **GPU-Accelerated**: All major operations use compute shaders

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd vk_symbiote

# Install dependencies
# Ubuntu/Debian
sudo apt-get install vulkan-dev cmake build-essential

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run examples
./vk_symbiote_example
./vk_symbiote_benchmark
```

### Basic Usage

```cpp
#include "vk_symbiote/VulkanSymbioteEngine.h"
#include "vk_symbiote/ConfigManager.h"

using namespace vk_symbiote;

int main(int argc, char* argv[]) {
    // Load configuration from command line or file
    ConfigManager::instance().load_from_args(argc, argv);
    
    // Initialize engine with model path
    VulkanSymbioteEngine engine("/path/to/model.gguf");
    
    // Generate text
    std::string result = engine.generate(
        "Once upon a time in a digital forest", 
        100,  // max tokens
        0.7f   // temperature
    );
    
    std::cout << "Generated: " << result << std::endl;
    
    return 0;
}
```

### Command Line Options

```
Usage: vk_symbiote_example [OPTIONS]

Model Configuration:
  --model <path>           Path to GGUF model file
  --config <path>          Configuration file path

Memory Configuration:
  --vram-gb <size>        VRAM budget in GB (default: 4)
  --ram-gb <size>         RAM budget in GB (default: 16)
  --prefetch-lookahead <n>  Prefetch lookahead (default: 3)
  --eviction-aggression <f>  Eviction aggression 0.0-1.0 (default: 0.7)

Performance:
  --workgroup-size <x> <y>   Workgroup size (default: 16 16)
  --disable-gpu             Use CPU fallback
  --fp32                    Use FP32 instead of FP16

Logging:
  --verbose, -v             Enable verbose logging
  --log-file <path>        Log to file
  --log-level <level>       Log level: DEBUG, INFO, WARNING, ERROR
```

## Configuration

### Configuration File Format

Configuration files use INI format:

```ini
[memory]
vram_budget_gb = 4
ram_budget_gb = 16
prefetch_lookahead = 3
eviction_aggression = 0.7
enable_compression = true
compression_algorithm = blosc2
max_packs_in_memory = 64

[performance]
enable_gpu = true
enable_profiling = false
workgroup_size_x = 16
workgroup_size_y = 16
use_subgroup_ops = true
use_fp16_math = true
scale_factor = 1.0

[logging]
log_level = INFO
log_to_file = false
log_file_path = vk_symbiote.log
log_performance = true
log_memory_usage = true
max_log_file_size_mb = 100

[model]
model_path = /path/to/model.gguf
```

### Environment Variables

```bash
export VK_SYMBIOTE_VRAM_GB=8
export VK_SYMBIOTE_RAM_GB=32
export VK_SYMBIOTE_PREFETCH=5
export VK_SYMBIOTE_CONFIG=/path/to/config.ini
```

## API Reference

### Core Classes

#### VulkanSymbioteEngine

```cpp
class VulkanSymbioteEngine {
public:
    VulkanSymbioteEngine(const Path& model_path);
    ~VulkanSymbioteEngine();
    
    // Text generation
    std::string generate(const std::string& prompt, 
                      uint32_t max_tokens = 256, 
                      float temperature = 0.7f);
    
    // Token operations
    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;
    
    // Performance monitoring
    const PerformanceMetrics& get_performance_metrics() const;
    void reset_performance_metrics();
};
```

#### PackManager

```cpp
class PackManager {
public:
    Expected<std::shared_ptr<NomadPack>> get_or_load_pack(uint64 pack_id);
    ExpectedVoid prefetch_packs(const std::vector<uint64>& pack_ids);
    ExpectedVoid evict_until(uint64 bytes_needed);
    
    // Memory statistics
    float vram_utilization() const;
    float ram_utilization() const;
    std::vector<uint64> get_loaded_pack_ids() const;
};
```

#### VitalityOracle

```cpp
class VitalityOracle {
public:
    VitalityScore score_pack(const PackMetadata& pack, 
                          const std::vector<uint32_t>& tokens,
                          uint32_t current_layer, 
                          const HardwareTelemetry& telemetry);
    
    std::vector<uint64> predict_next_packs(const std::vector<uint32_t>& tokens,
                                           uint32_t current_layer, 
                                           uint32 lookahead = 3, 
                                           uint32 count = 8);
    
    // Performance feedback
    void record_access(uint64 pack_id, bool was_used, float predicted_score);
    void update_model();
};
```

#### ShaderRuntime

```cpp
class ShaderRuntime {
public:
    struct ShaderSpecialization {
        uint32_t workgroup_size_x = 16;
        uint32_t workgroup_size_y = 16;
        bool use_subgroup_ops = true;
        bool use_fp16_math = true;
    };
    
    VkPipeline get_fused_matmul_rope_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_attention_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_feedforward_pipeline(const ShaderSpecialization& spec);
    VkPipeline get_rms_norm_pipeline(const ShaderSpecialization& spec);
    
    void dispatch_compute(VkPipeline pipeline, 
                        VkDescriptorSet descriptor_set,
                        uint32_t group_count_x,
                        uint32_t group_count_y = 1,
                        uint32_t group_count_z = 1);
};
```

## Performance Optimization

### Memory Optimization

1. **Pack Configuration**: Adjust pack sizes based on model and hardware
2. **Prefetch Strategy**: Optimize lookahead for your usage pattern
3. **Eviction Aggression**: Balance memory pressure vs. cache hits
4. **Compression**: Enable lossless compression for larger models

### GPU Optimization

1. **Workgroup Sizing**: Match your GPU's compute units
2. **FP16 Math**: Enable for better performance on supported hardware
3. **Subgroup Operations**: Use for parallel reduction
4. **Batch Sizing**: Process multiple tokens when possible

### Scheduling Optimization

```cpp
// Example: Optimize for throughput
ConfigManager::instance().performance().prefetch_lookahead = 5;
ConfigManager::instance().performance().workgroup_size_x = 32;
ConfigManager::instance().memory().eviction_aggression = 0.5f;
```

### Performance Monitoring

```cpp
// Enable performance profiling
ConfigManager::instance().performance().enable_profiling = true;

// Monitor in real-time
const auto& metrics = engine.get_performance_metrics();
std::cout << "Tokens/sec: " << metrics.average_tokens_per_second << std::endl;
std::cout << "GPU utilization: " << metrics.gpu_utilization_percent << "%" << std::endl;
std::cout << "Memory usage: " << (metrics.total_gpu_time_ns / 1e9f) << " MB/s" << std::endl;
```

## Troubleshooting

### Common Issues

#### Out of Memory
```
# Reduce memory budgets
--vram-gb 2
--ram-gb 8

# Increase eviction aggression
--eviction-aggression 0.9

# Disable compression
--no-compression
```

#### Slow Performance
```
# Disable GPU (use CPU fallback)
--disable-gpu

# Increase workgroup size
--workgroup-size 32 32

# Disable FP16
--fp32

# Disable subgroups
--no-subgroups
```

#### Vulkan Errors
```
# Enable verbose logging
--verbose

# Check GPU support
vulkaninfo --summary

# Force integrated GPU
--force-integrated-gpu
```

#### Model Loading Issues
```
# Check model format
file model.gguf

# Verify model path
ls -la /path/to/model.gguf

# Check available memory
free -h

# Enable debug logging
--log-level DEBUG
```

## Developer Guide

### Adding New Compute Kernels

1. Create GLSL shader in `shaders/` directory
2. Use specialization constants for runtime parameters
3. Implement proper memory barriers
4. Add subgroup optimizations where possible

### Extending Pack Types

1. Update `PackType` enum in `Common.h`
2. Add metadata to `PackMetadata` struct
3. Update pack generation logic in `GGUFLoader.cpp`
4. Implement specialized loading/unloading

### Custom Compression

1. Implement `CompressionBackend` interface
2. Add to build system in `CMakeLists.txt`
3. Register in `Compression.cpp`
4. Update pack manager to use new algorithm

### Performance Profiling

```cpp
// Enable profiling in code
ConfigManager::instance().performance().enable_profiling = true;

// Access timing data
const auto& metrics = engine.get_performance_metrics();
uint64_t avg_inference_time = metrics.total_inference_time_ns / metrics.total_inferences;
```

### Memory Debugging

```cpp
// Enable memory usage logging
ConfigManager::instance().logging().log_memory_usage = true;

// Monitor pack statistics
const auto& vram_stats = pack_manager.get_vram_stats();
const auto& ram_stats = pack_manager.get_ram_stats();

// Force memory pressure
ConfigManager::instance().memory().vram_budget_gb = 1;
```

## License

[License details to be added]

## Contributing

This is a research-grade prototype. Key areas for improvement:

- Additional model architectures (Mistral, Qwen, etc.)
- Multi-GPU support
- Advanced compression algorithms
- CPU fallback implementations
- Production-ready error handling

## Acknowledgments

Built with:
- Vulkan 1.3+ compute shaders
- VMA memory allocation
- GGUF model format support
- Modern C++20 features
- Blosc2 compression library