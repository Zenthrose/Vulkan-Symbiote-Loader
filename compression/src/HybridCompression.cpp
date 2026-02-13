#include "HybridCompression.h"
#include <cstring>
#include <algorithm>
#include <chrono>
#include <regex>

namespace compression {

// CRC32 lookup table for integrity checking
static const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

HybridCompression::HybridCompression() = default;
HybridCompression::~HybridCompression() = default;

bool HybridCompression::initialize() {
    // Initialize all backends
    bool success = true;
    
    success &= Blosc2Compression::initialize();
    
    // Initialize different Blosc2 configurations
    blosc2_lz4_ = std::make_unique<Blosc2Compression>();
    blosc2_zstd_ = std::make_unique<Blosc2Compression>();
    
    // Initialize ZFP backends if available
    if (ZFPCompression::initialize()) {
        zfp_rate_ = std::make_unique<ZFPCompression>(ZFPParams::rate_mode(4.0));
        zfp_precision_ = std::make_unique<ZFPCompression>(ZFPParams::precision_mode(12));
        zfp_accuracy_ = std::make_unique<ZFPCompression>(ZFPParams::accuracy_mode(0.001));
    }
    
    // Set default profiles
    set_profile("attention.*k", CompressionProfile::lossy_size());      // KV cache can be lossy
    set_profile("attention.*v", CompressionProfile::lossy_size());
    set_profile("weight", CompressionProfile::lossless_fast());         // Weights must be lossless
    set_profile("embed", CompressionProfile::lossless_max());           // Embeddings once, compress well
    set_profile("norm", CompressionProfile::lossless_fast());           // Norms are small, fast is fine
    
    return success;
}

void HybridCompression::shutdown() {
    blosc2_lz4_.reset();
    blosc2_zstd_.reset();
    zfp_rate_.reset();
    zfp_precision_.reset();
    zfp_accuracy_.reset();
    ZFPCompression::shutdown();
}

bool HybridCompression::compress(const void* src, size_t src_size,
                                 std::vector<uint8_t>& out,
                                 CompressionType type,
                                 const TensorMetadata* metadata) {
    if (!src || src_size == 0) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Auto-select compression type if requested
    if (type == CompressionType::AUTO && metadata) {
        auto profile = get_profile(metadata->name, metadata);
        type = profile.type;
    }
    
    // Reserve space for header + compressed data
    out.reserve(sizeof(CompressionHeader) + src_size);
    out.resize(sizeof(CompressionHeader));
    
    std::vector<uint8_t> compressed_data;
    bool success = false;
    
    // Compress based on type
    switch (type) {
        case CompressionType::NONE:
            // Passthrough - just copy data
            compressed_data.resize(src_size);
            std::memcpy(compressed_data.data(), src, src_size);
            success = true;
            break;
            
        case CompressionType::BLOSC2_LZ4:
        case CompressionType::BLOSC2_LZ4HC:
            if (blosc2_lz4_) {
                success = blosc2_lz4_->compress(src, src_size, compressed_data);
            }
            break;
            
        case CompressionType::BLOSC2_ZSTD:
            if (blosc2_zstd_) {
                success = blosc2_zstd_->compress(src, src_size, compressed_data);
            }
            break;
            
        case CompressionType::ZFP_RATE:
            if (zfp_rate_ && metadata) {
                success = compress_with_zfp(src, src_size, compressed_data, 
                                           zfp_rate_->get_params(), metadata->dimensions);
            }
            break;
            
        case CompressionType::ZFP_PRECISION:
            if (zfp_precision_ && metadata) {
                success = compress_with_zfp(src, src_size, compressed_data,
                                           zfp_precision_->get_params(), metadata->dimensions);
            }
            break;
            
        case CompressionType::ZFP_ACCURACY:
            if (zfp_accuracy_ && metadata) {
                success = compress_with_zfp(src, src_size, compressed_data,
                                           zfp_accuracy_->get_params(), metadata->dimensions);
            }
            break;
            
        default:
            // Fallback to Blosc2 LZ4
            if (blosc2_lz4_) {
                success = blosc2_lz4_->compress(src, src_size, compressed_data);
            }
            break;
    }
    
    if (!success || compressed_data.empty()) {
        return false;
    }
    
    // Write header
    if (!write_header(out, type, src_size, compressed_data.size())) {
        return false;
    }
    
    // Append compressed data
    out.insert(out.end(), compressed_data.begin(), compressed_data.end());
    
    // Update stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    stats_.total_compressed++;
    double ratio = static_cast<double>(src_size) / compressed_data.size();
    stats_.avg_compression_ratio = (stats_.avg_compression_ratio * (stats_.total_compressed - 1) + ratio) 
                                   / stats_.total_compressed;
    stats_.avg_compress_time_ms = (stats_.avg_compress_time_ms * (stats_.total_compressed - 1) 
                                   + duration.count() / 1000.0) / stats_.total_compressed;
    
    return true;
}

bool HybridCompression::decompress(const uint8_t* src, size_t src_size,
                                   size_t decompressed_size, void* dst) {
    if (!src || src_size < sizeof(CompressionHeader) || !dst) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Read header
    CompressionHeader header;
    if (!read_header(src, src_size, header)) {
        return false;
    }
    
    // Verify CRC32
    const uint8_t* data_start = src + sizeof(CompressionHeader);
    size_t data_size = header.compressed_size;
    uint32_t calculated_crc = calculate_crc32(data_start, data_size);
    if (calculated_crc != header.crc32) {
        // CRC mismatch - data corruption detected
        return false;
    }
    
    bool success = false;
    
    // Decompress based on method
    switch (static_cast<CompressionType>(header.method)) {
        case CompressionType::NONE:
            std::memcpy(dst, data_start, decompressed_size);
            success = true;
            break;
            
        case CompressionType::BLOSC2_LZ4:
        case CompressionType::BLOSC2_LZ4HC:
        case CompressionType::BLOSC2_ZSTD:
            if (blosc2_lz4_) {
                success = blosc2_lz4_->decompress(data_start, data_size, decompressed_size, dst);
            }
            break;
            
        case CompressionType::ZFP_RATE:
        case CompressionType::ZFP_PRECISION:
        case CompressionType::ZFP_ACCURACY:
            if (zfp_rate_) {
                success = zfp_rate_->decompress(data_start, data_size, decompressed_size, dst);
            }
            break;
            
        default:
            return false;
    }
    
    if (success) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        stats_.total_decompressed++;
        stats_.avg_decompress_time_ms = (stats_.avg_decompress_time_ms * (stats_.total_decompressed - 1)
                                         + duration.count() / 1000.0) / stats_.total_decompressed;
    }
    
    return success;
}

void HybridCompression::set_profile(const std::string& name_pattern, const CompressionProfile& profile) {
    profiles_.emplace_back(name_pattern, profile);
}

CompressionProfile HybridCompression::get_profile(const std::string& tensor_name,
                                                  const TensorMetadata* metadata) const {
    // Check registered patterns (in reverse order - last match wins)
    for (auto it = profiles_.rbegin(); it != profiles_.rend(); ++it) {
        try {
            std::regex pattern(it->first);
            if (std::regex_search(tensor_name, pattern)) {
                return it->second;
            }
        } catch (...) {
            // Invalid regex, skip
            continue;
        }
    }
    
    // Auto-select based on metadata
    if (metadata) {
        return auto_select_profile(*metadata);
    }
    
    // Default: fast lossless
    return CompressionProfile::lossless_fast();
}

CompressionProfile HybridCompression::auto_select_profile(const TensorMetadata& metadata) {
    // KV cache: Use lossy compression for memory savings
    if (metadata.is_kv_cache) {
        return CompressionProfile::lossy_size();
    }
    
    // Weights: Must be lossless
    if (metadata.is_weight) {
        return CompressionProfile::lossless_fast();
    }
    
    // Small tensors: Fast compression
    if (metadata.total_elements < 1024) {
        return CompressionProfile::lossless_fast();
    }
    
    // Large tensors with smooth data: Use ZFP
    if (metadata.total_elements > 100000 && metadata.dynamic_range < 1000.0f) {
        return CompressionProfile::lossy_quality();
    }
    
    // Default: balanced
    return CompressionProfile{CompressionType::BLOSC2_LZ4, ZFPParams(), 5, true};
}

CompressionType HybridCompression::detect_optimal_type(const TensorMetadata& metadata) {
    return auto_select_profile(metadata).type;
}

bool HybridCompression::should_use_lossy(const TensorMetadata& metadata) {
    return metadata.is_kv_cache || metadata.is_activation;
}

float HybridCompression::estimate_compression_ratio(CompressionType type, const TensorMetadata& metadata) {
    switch (type) {
        case CompressionType::NONE:
            return 1.0f;
        case CompressionType::BLOSC2_LZ4:
            return 1.5f;
        case CompressionType::BLOSC2_LZ4HC:
            return 2.0f;
        case CompressionType::BLOSC2_ZSTD:
            return 2.5f;
        case CompressionType::ZFP_RATE:
            return 32.0f / 4.0f;  // 4 bits per value = 8x
        case CompressionType::ZFP_PRECISION:
            return 4.0f;  // ~4x for 8-bit precision
        case CompressionType::ZFP_ACCURACY:
            return 3.0f;  // ~3x depending on tolerance
        default:
            return 1.5f;
    }
}

bool HybridCompression::compress_with_blosc2(const void* src, size_t src_size,
                                             std::vector<uint8_t>& out, int clevel) {
    if (!blosc2_lz4_) {
        return false;
    }
    return blosc2_lz4_->compress(src, src_size, out);
}

bool HybridCompression::compress_with_zfp(const void* src, size_t src_size,
                                          std::vector<uint8_t>& out, const ZFPParams& params,
                                          const std::vector<size_t>& dimensions) {
    if (!zfp_rate_) {
        return false;
    }
    
    // Select appropriate ZFP compressor
    ZFPCompression* compressor = nullptr;
    switch (params.mode) {
        case ZFPMode::RATE:
            compressor = zfp_rate_.get();
            break;
        case ZFPMode::PRECISION:
            compressor = zfp_precision_.get();
            break;
        case ZFPMode::ACCURACY:
            compressor = zfp_accuracy_.get();
            break;
        default:
            compressor = zfp_rate_.get();
            break;
    }
    
    if (!compressor) {
        return false;
    }
    
    // Compress based on dimensionality
    const float* data = static_cast<const float*>(src);
    bool success = false;
    
    if (dimensions.size() == 1) {
        success = compressor->compress_1d(data, dimensions[0], out);
    } else if (dimensions.size() == 2) {
        success = compressor->compress_2d(data, dimensions[0], dimensions[1], out);
    } else if (dimensions.size() == 3) {
        success = compressor->compress_3d(data, dimensions[0], dimensions[1], dimensions[2], out);
    } else {
        // Fallback to 1D
        size_t total = src_size / sizeof(float);
        success = compressor->compress_1d(data, total, out);
    }
    
    return success;
}

bool HybridCompression::write_header(std::vector<uint8_t>& out, CompressionType type,
                                     size_t uncompressed_size, size_t compressed_size) {
    if (out.size() < sizeof(CompressionHeader)) {
        return false;
    }
    
    CompressionHeader* header = reinterpret_cast<CompressionHeader*>(out.data());
    header->magic = HYBRID_MAGIC;
    header->version = HEADER_VERSION;
    header->method = static_cast<uint8_t>(type);
    header->flags = 0;
    header->uncompressed_size = uncompressed_size;
    header->compressed_size = compressed_size;
    header->crc32 = 0;  // Will be calculated after data is appended
    
    return true;
}

bool HybridCompression::read_header(const uint8_t* src, size_t src_size, CompressionHeader& header) {
    if (src_size < sizeof(CompressionHeader)) {
        return false;
    }
    
    std::memcpy(&header, src, sizeof(CompressionHeader));
    
    // Verify magic
    if (header.magic != HYBRID_MAGIC) {
        return false;
    }
    
    // Verify version
    if (header.version != HEADER_VERSION) {
        return false;
    }
    
    return true;
}

uint32_t HybridCompression::calculate_crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; ++i) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

// Global instance
HybridCompression& get_global_hybrid_compression() {
    static HybridCompression instance;
    return instance;
}

} // namespace compression
