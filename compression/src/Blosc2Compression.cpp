#include "Blosc2Compression.h"
#include <cstring>
#include <vector>

#ifdef BLOSC2_ENABLED
// Real Blosc2 integration path
// This file assumes Blosc2 headers are available: include <blosc2.h>
// The actual implementation depends on presence of Blosc2 in build environment.
// If Blosc2 is not available, this file will be compiled with stub path in the header.
#include <blosc2.h>

namespace compression {
Blosc2Compression::Blosc2Compression() {
}

Blosc2Compression::~Blosc2Compression() {
}

bool Blosc2Compression::initialize() {
    // Initialize Blosc2 once per process
    // Best-effort; return true on success or if already initialized
    static bool initialized = false;
    if (initialized) return true;
    int rc = blosc2_init();
    initialized = (rc == 0);
    return initialized;
}

bool Blosc2Compression::compress(const void* src, size_t src_size, std::vector<uint8_t>& out) {
    if (!src || src_size == 0) { out.clear(); return true; }
    
    // Use Blosc2 for optimal compression with proper parameters
    int compression_level = 5; // Balanced compression level
    int typesize = 4; // Assuming float32 data
    int64_t nelems = static_cast<int64_t>(src_size / typesize);
    
    // Estimate compressed size with some overhead
    size_t compressed_bound = blosc2_csize(typesize, nelems, compression_level);
    out.resize(compressed_bound + BLOSC2_MAX_OVERHEAD);
    
    // Compress using default codec (LZ4) with shuffle
    int64_t compressed_bytes = blosc2_compress(src, nelems, typesize, out.data(), 
                                              compressed_bound + BLOSC2_MAX_OVERHEAD,
                                              compression_level, BLOSC_SHUFFLE, BLOSC_LZ4);
    
    if (compressed_bytes <= 0) {
        // Compression failed, fallback to raw copy
        out.resize(src_size);
        std::memcpy(out.data(), src, src_size);
        return true;
    }
    
    out.resize(static_cast<size_t>(compressed_bytes));
    return true;
}

bool Blosc2Compression::decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) {
    if (!src_data || src_size == 0 || !dst) return false;
    
    int64_t decompressed_bytes = blosc2_decompress(src_data, src_size, dst, 
                                                  decompressed_size, 0);
    
    return (decompressed_bytes == static_cast<int64_t>(decompressed_size));
}
}
#else
// Enhanced stub implementations when BLOSC2 is not available
namespace compression {
static std::vector<uint8_t> compression_buffer_; // Simple compression cache

Blosc2Compression::Blosc2Compression() {
    // Initialize simple RLE compression for stub implementation
    compression_buffer_.reserve(1024 * 1024); // 1MB cache
}

Blosc2Compression::~Blosc2Compression() {
    compression_buffer_.clear();
}

bool Blosc2Compression::compress(const void* src, size_t src_size, std::vector<uint8_t>& out) {
    if (!src || src_size == 0) { 
        out.clear(); 
        return true; 
    }
    
    // Simple RLE compression for fallback
    const uint8_t* data = static_cast<const uint8_t*>(src);
    compression_buffer_.clear();
    
    for (size_t i = 0; i < src_size; ) {
        uint8_t value = data[i];
        size_t count = 1;
        
        // Count consecutive identical bytes
        while (i + count < src_size && data[i + count] == value && count < 255) {
            count++;
        }
        
        // Write RLE entry: value followed by count
        compression_buffer_.push_back(value);
        compression_buffer_.push_back(static_cast<uint8_t>(count));
        
        i += count;
    }
    
    out = compression_buffer_;
    return true;
}

bool Blosc2Compression::decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) {
    if (!src_data || src_size == 0 || !dst) return false;
    
    uint8_t* output = static_cast<uint8_t*>(dst);
    size_t output_idx = 0;
    size_t input_idx = 0;
    
    // Simple RLE decompression
    while (input_idx < src_size && output_idx < decompressed_size) {
        if (input_idx + 1 >= src_size) break;
        
        uint8_t value = src_data[input_idx++];
        uint8_t count = src_data[input_idx++];
        
        // Clamp count to avoid buffer overflow
        size_t actual_count = std::min(static_cast<size_t>(count), 
                                    decompressed_size - output_idx);
        
        // Write decompressed data
        for (size_t i = 0; i < actual_count; ++i) {
            if (output_idx < decompressed_size) {
                output[output_idx++] = value;
            }
        }
    }
    
    return (output_idx == decompressed_size);
}

bool Blosc2Compression::initialize() { 
    return true; 
}
}
#endif
