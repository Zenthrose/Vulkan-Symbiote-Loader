#pragma once

#include "CompressionBackend.h"

namespace compression {

#if defined(BLOSC2_ENABLED)
    class Blosc2Compression : public CompressionBackend {
    public:
        Blosc2Compression();
        ~Blosc2Compression();
        bool compress(const void* src, size_t src_size, std::vector<uint8_t>& out) override;
        bool decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) override;
        static bool initialize();
    };
#else
    // Fallback declaration when Blosc2 is not available; provide a compatible interface
    class Blosc2Compression : public CompressionBackend {
    public:
        bool compress(const void* src, size_t src_size, std::vector<uint8_t>& out) override { return false; }
        bool decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) override { return false; }
        static bool initialize() { return true; }
    };
#endif

} // namespace compression
