#include "ZFPCompression.h"
#include <cstring>
#include <algorithm>

#if defined(ZFP_ENABLED)
#include <zfp.h>

namespace compression {

// Static initialization
static bool zfp_initialized = false;

bool ZFPCompression::initialize() {
    if (!zfp_initialized) {
        // ZFP doesn't require explicit initialization
        zfp_initialized = true;
    }
    return zfp_initialized;
}

void ZFPCompression::shutdown() {
    zfp_initialized = false;
}

ZFPCompression::ZFPCompression(const ZFPParams& params) : params_(params) {
    initialize_stream();
}

ZFPCompression::~ZFPCompression() = default;

void ZFPCompression::ZFPStreamDeleter::operator()(zfp_stream* stream) {
    if (stream) {
        zfp_stream_close(stream);
    }
}

void ZFPCompression::ZFPFieldDeleter::operator()(zfp_field* field) {
    if (field) {
        zfp_field_free(field);
    }
}

bool ZFPCompression::initialize_stream() {
    stream_.reset(zfp_stream_open(nullptr));
    if (!stream_) {
        return false;
    }
    
    // Configure compression mode
    switch (params_.mode) {
        case ZFPMode::RATE:
            zfp_stream_set_rate(stream_.get(), params_.rate, zfp_type_float, 1, 0);
            break;
        case ZFPMode::PRECISION:
            zfp_stream_set_precision(stream_.get(), params_.precision);
            break;
        case ZFPMode::ACCURACY:
            zfp_stream_set_accuracy(stream_.get(), params_.accuracy);
            break;
        case ZFPMode::EXPERT:
            // Expert mode requires manual setting
            zfp_stream_set_rate(stream_.get(), params_.rate, zfp_type_float, 1, 0);
            break;
    }
    
    return true;
}

bool ZFPCompression::compress(const void* src, size_t src_size, std::vector<uint8_t>& out) {
    if (!src || src_size == 0) {
        return false;
    }
    
    // Assume 1D float array for generic interface
    size_t num_elements = src_size / sizeof(float);
    return compress_1d(static_cast<const float*>(src), num_elements, out);
}

bool ZFPCompression::compress_1d(const float* src, size_t n, std::vector<uint8_t>& out) {
    if (!src || n == 0) {
        return false;
    }
    
    // Create field
    field_.reset(zfp_field_1d(const_cast<float*>(src), zfp_type_float, n));
    if (!field_) {
        return false;
    }
    
    // Allocate buffer for compressed data
    size_t max_compressed_size = zfp_stream_maximum_size(stream_.get(), field_.get());
    out.resize(max_compressed_size);
    
    // Open bitstream for writing
    bitstream* stream = stream_open(out.data(), max_compressed_size);
    if (!stream) {
        return false;
    }
    
    zfp_stream_set_bit_stream(stream_.get(), stream);
    zfp_stream_rewind(stream_.get());
    
    // Compress
    size_t compressed_size = zfp_compress(stream_.get(), field_.get());
    stream_close(stream);
    
    if (compressed_size == 0) {
        return false;
    }
    
    out.resize(compressed_size);
    last_compression_ratio_ = static_cast<double>(n * sizeof(float)) / compressed_size;
    
    return true;
}

bool ZFPCompression::compress_2d(const float* src, size_t nx, size_t ny, std::vector<uint8_t>& out) {
    if (!src || nx == 0 || ny == 0) {
        return false;
    }
    
    field_.reset(zfp_field_2d(const_cast<float*>(src), zfp_type_float, nx, ny));
    if (!field_) {
        return false;
    }
    
    size_t max_compressed_size = zfp_stream_maximum_size(stream_.get(), field_.get());
    out.resize(max_compressed_size);
    
    bitstream* stream = stream_open(out.data(), max_compressed_size);
    if (!stream) {
        return false;
    }
    
    zfp_stream_set_bit_stream(stream_.get(), stream);
    zfp_stream_rewind(stream_.get());
    
    size_t compressed_size = zfp_compress(stream_.get(), field_.get());
    stream_close(stream);
    
    if (compressed_size == 0) {
        return false;
    }
    
    out.resize(compressed_size);
    last_compression_ratio_ = static_cast<double>(nx * ny * sizeof(float)) / compressed_size;
    
    return true;
}

bool ZFPCompression::compress_3d(const float* src, size_t nx, size_t ny, size_t nz, std::vector<uint8_t>& out) {
    if (!src || nx == 0 || ny == 0 || nz == 0) {
        return false;
    }
    
    field_.reset(zfp_field_3d(const_cast<float*>(src), zfp_type_float, nx, ny, nz));
    if (!field_) {
        return false;
    }
    
    size_t max_compressed_size = zfp_stream_maximum_size(stream_.get(), field_.get());
    out.resize(max_compressed_size);
    
    bitstream* stream = stream_open(out.data(), max_compressed_size);
    if (!stream) {
        return false;
    }
    
    zfp_stream_set_bit_stream(stream_.get(), stream);
    zfp_stream_rewind(stream_.get());
    
    size_t compressed_size = zfp_compress(stream_.get(), field_.get());
    stream_close(stream);
    
    if (compressed_size == 0) {
        return false;
    }
    
    out.resize(compressed_size);
    last_compression_ratio_ = static_cast<double>(nx * ny * nz * sizeof(float)) / compressed_size;
    
    return true;
}

bool ZFPCompression::decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) {
    if (!src_data || src_size == 0 || !dst) {
        return false;
    }
    
    size_t num_elements = decompressed_size / sizeof(float);
    return decompress_1d(src_data, src_size, static_cast<float*>(dst), num_elements);
}

bool ZFPCompression::decompress_1d(const uint8_t* src, size_t src_size, float* dst, size_t n) {
    if (!src || src_size == 0 || !dst || n == 0) {
        return false;
    }
    
    field_.reset(zfp_field_1d(dst, zfp_type_float, n));
    if (!field_) {
        return false;
    }
    
    bitstream* stream = stream_open(const_cast<uint8_t*>(src), src_size);
    if (!stream) {
        return false;
    }
    
    zfp_stream_set_bit_stream(stream_.get(), stream);
    zfp_stream_rewind(stream_.get());
    
    int success = zfp_decompress(stream_.get(), field_.get());
    stream_close(stream);
    
    return success != 0;
}

bool ZFPCompression::decompress_2d(const uint8_t* src, size_t src_size, float* dst, size_t nx, size_t ny) {
    if (!src || src_size == 0 || !dst || nx == 0 || ny == 0) {
        return false;
    }
    
    field_.reset(zfp_field_2d(dst, zfp_type_float, nx, ny));
    if (!field_) {
        return false;
    }
    
    bitstream* stream = stream_open(const_cast<uint8_t*>(src), src_size);
    if (!stream) {
        return false;
    }
    
    zfp_stream_set_bit_stream(stream_.get(), stream);
    zfp_stream_rewind(stream_.get());
    
    int success = zfp_decompress(stream_.get(), field_.get());
    stream_close(stream);
    
    return success != 0;
}

bool ZFPCompression::decompress_3d(const uint8_t* src, size_t src_size, float* dst, size_t nx, size_t ny, size_t nz) {
    if (!src || src_size == 0 || !dst || nx == 0 || ny == 0 || nz == 0) {
        return false;
    }
    
    field_.reset(zfp_field_3d(dst, zfp_type_float, nx, ny, nz));
    if (!field_) {
        return false;
    }
    
    bitstream* stream = stream_open(const_cast<uint8_t*>(src), src_size);
    if (!stream) {
        return false;
    }
    
    zfp_stream_set_bit_stream(stream_.get(), stream);
    zfp_stream_rewind(stream_.get());
    
    int success = zfp_decompress(stream_.get(), field_.get());
    stream_close(stream);
    
    return success != 0;
}

} // namespace compression

#else
// Fallback implementation when ZFP is not available
namespace compression {
    bool ZFPCompression::initialize() { return false; }
    void ZFPCompression::shutdown() {}
} // namespace compression
#endif
