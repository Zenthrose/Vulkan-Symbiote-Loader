#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace compression {

// Abstract lossless codec backend interface
class CompressionBackend {
public:
    virtual ~CompressionBackend() = default;
    // Compress src of size src_size into out (byte vector). Returns true on success.
    virtual bool compress(const void* src, size_t src_size, std::vector<uint8_t>& out) = 0;
    // Decompress src_data (compressed) of size src_size into dst of size decompressed_size.
    virtual bool decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) = 0;
};

} // namespace compression
