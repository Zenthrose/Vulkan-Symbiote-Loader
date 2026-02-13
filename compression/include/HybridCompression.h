#pragma once

#include "CompressionBackend.h"
#include "Blosc2Compression.h"
#include "ZFPCompression.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace compression {

// Compression strategy types
enum class CompressionType {
    NONE,               // No compression (passthrough)
    BLOSC2_LZ4,         // Fast, lossless
    BLOSC2_LZ4HC,       // Better ratio than LZ4
    BLOSC2_ZSTD,        // Best ratio, slower
    ZFP_RATE,           // Fixed rate (bits per value)
    ZFP_PRECISION,      // Fixed precision
    ZFP_ACCURACY,       // Fixed accuracy
    AUTO                // Automatic selection based on data
};

// Tensor metadata for compression selection
struct TensorMetadata {
    std::string name;
    std::vector<size_t> dimensions;
    size_t total_elements;
    bool is_weight;         // Model weights (need lossless)
    bool is_activation;     // Activations (can be lossy)
    bool is_kv_cache;       // KV cache (can be lossy for memory savings)
    float min_value;
    float max_value;
    float dynamic_range;
};

// Compression profile for a tensor type
struct CompressionProfile {
    CompressionType type;
    ZFPParams zfp_params;
    int blosc_level;        // 1-9 for Blosc2
    bool use_shuffling;     // Byte shuffling for better compression
    
    // Default profiles
    static CompressionProfile lossless_fast() {
        return {CompressionType::BLOSC2_LZ4, ZFPParams(), 1, true};
    }
    
    static CompressionProfile lossless_max() {
        return {CompressionType::BLOSC2_ZSTD, ZFPParams(), 9, true};
    }
    
    static CompressionProfile lossy_quality() {
        return {CompressionType::ZFP_PRECISION, ZFPParams::precision_mode(12), 0, false};
    }
    
    static CompressionProfile lossy_size() {
        return {CompressionType::ZFP_RATE, ZFPParams::rate_mode(4.0), 0, false};
    }
};

// Hybrid compression manager
class HybridCompression {
public:
    HybridCompression();
    ~HybridCompression();
    
    // Initialize all compression backends
    bool initialize();
    void shutdown();
    
    // Compress data with automatic or manual method selection
    bool compress(const void* src, size_t src_size, 
                  std::vector<uint8_t>& out,
                  CompressionType type = CompressionType::AUTO,
                  const TensorMetadata* metadata = nullptr);
    
    // Decompress data (auto-detects method from header)
    bool decompress(const uint8_t* src, size_t src_size, 
                    size_t decompressed_size, void* dst);
    
    // Set compression profile for a tensor name pattern
    void set_profile(const std::string& name_pattern, const CompressionProfile& profile);
    
    // Get profile for a tensor
    CompressionProfile get_profile(const std::string& tensor_name, 
                                   const TensorMetadata* metadata = nullptr) const;
    
    // Automatic profile selection based on tensor characteristics
    static CompressionProfile auto_select_profile(const TensorMetadata& metadata);
    
    // Compression statistics
    struct Stats {
        size_t total_compressed = 0;
        size_t total_decompressed = 0;
        double avg_compression_ratio = 0.0;
        double avg_compress_time_ms = 0.0;
        double avg_decompress_time_ms = 0.0;
    };
    
    Stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats(); }
    
    // Utility functions
    static CompressionType detect_optimal_type(const TensorMetadata& metadata);
    static bool should_use_lossy(const TensorMetadata& metadata);
    static float estimate_compression_ratio(CompressionType type, const TensorMetadata& metadata);

private:
    // Backend instances
    std::unique_ptr<Blosc2Compression> blosc2_lz4_;
    std::unique_ptr<Blosc2Compression> blosc2_zstd_;
    std::unique_ptr<ZFPCompression> zfp_rate_;
    std::unique_ptr<ZFPCompression> zfp_precision_;
    std::unique_ptr<ZFPCompression> zfp_accuracy_;
    
    // Profile registry
    std::vector<std::pair<std::string, CompressionProfile>> profiles_;
    
    // Statistics
    mutable Stats stats_;
    
    // Helper methods
    bool compress_with_blosc2(const void* src, size_t src_size, 
                              std::vector<uint8_t>& out, int clevel);
    bool compress_with_zfp(const void* src, size_t src_size,
                           std::vector<uint8_t>& out, const ZFPParams& params,
                           const std::vector<size_t>& dimensions);
    
    // Header format for compressed data
    struct CompressionHeader {
        uint32_t magic;          // 'HYBR' = 0x48595242
        uint8_t version;         // Format version
        uint8_t method;          // CompressionType
        uint16_t flags;          // Additional flags
        uint64_t uncompressed_size;
        uint64_t compressed_size;
        uint32_t crc32;          // Data integrity check
    };
    
    static constexpr uint32_t HYBRID_MAGIC = 0x48595242;  // 'HYBR'
    static constexpr uint8_t HEADER_VERSION = 1;
    
    bool write_header(std::vector<uint8_t>& out, CompressionType type,
                      size_t uncompressed_size, size_t compressed_size);
    bool read_header(const uint8_t* src, size_t src_size, CompressionHeader& header);
    uint32_t calculate_crc32(const uint8_t* data, size_t size);
};

// Global hybrid compression instance
HybridCompression& get_global_hybrid_compression();

} // namespace compression
