#pragma once

#include "CompressionBackend.h"
#include <cstddef>
#include <vector>
#include <memory>

// Include or forward declare ZFP library types
#if defined(ZFP_ENABLED)
#include <zfp.h>
#else
struct zfp_stream;
struct zfp_field;
#endif

namespace compression {

// ZFP compression modes
enum class ZFPMode {
    RATE,       // Fixed rate (bits per value)
    PRECISION,  // Fixed precision (bits of accuracy)
    ACCURACY,   // Fixed accuracy (absolute error tolerance)
    EXPERT      // Expert mode (tolerance + rate + precision)
};

// ZFP compression parameters
struct ZFPParams {
    ZFPMode mode = ZFPMode::RATE;
    union {
        double rate;       // Bits per value (e.g., 4.0 for 4 bits)
        double precision;  // Bit precision (e.g., 12 for 12 bits)
        double accuracy;   // Absolute error tolerance (e.g., 0.001)
    };
    
    ZFPParams() : rate(8.0) {}  // Default: 8 bits per value (50% compression)
    
    static ZFPParams rate_mode(double bits_per_value) {
        ZFPParams p;
        p.mode = ZFPMode::RATE;
        p.rate = bits_per_value;
        return p;
    }
    
    static ZFPParams precision_mode(double bits) {
        ZFPParams p;
        p.mode = ZFPMode::PRECISION;
        p.precision = bits;
        return p;
    }
    
    static ZFPParams accuracy_mode(double tolerance) {
        ZFPParams p;
        p.mode = ZFPMode::ACCURACY;
        p.accuracy = tolerance;
        return p;
    }
};

#if defined(ZFP_ENABLED)
    class ZFPCompression : public CompressionBackend {
    public:
        explicit ZFPCompression(const ZFPParams& params = ZFPParams());
        ~ZFPCompression();
        
        // CompressionBackend interface
        bool compress(const void* src, size_t src_size, std::vector<uint8_t>& out) override;
        bool decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) override;
        
        // Static initialization
        static bool initialize();
        static void shutdown();
        
        // ZFP-specific methods
        bool compress_1d(const float* src, size_t n, std::vector<uint8_t>& out);
        bool compress_2d(const float* src, size_t nx, size_t ny, std::vector<uint8_t>& out);
        bool compress_3d(const float* src, size_t nx, size_t ny, size_t nz, std::vector<uint8_t>& out);
        
        bool decompress_1d(const uint8_t* src, size_t src_size, float* dst, size_t n);
        bool decompress_2d(const uint8_t* src, size_t src_size, float* dst, size_t nx, size_t ny);
        bool decompress_3d(const uint8_t* src, size_t src_size, float* dst, size_t nx, size_t ny, size_t nz);
        
        // Get compression ratio from last operation
        double get_last_compression_ratio() const { return last_compression_ratio_; }
        
        // Update parameters
        void set_params(const ZFPParams& params) { params_ = params; }
        const ZFPParams& get_params() const { return params_; }
        
    private:
        ZFPParams params_;
        double last_compression_ratio_ = 0.0;
        
        // ZFP stream and field (opaque pointers to avoid header dependency)
        struct ZFPStreamDeleter {
            void operator()(zfp_stream* stream);
        };
        struct ZFPFieldDeleter {
            void operator()(zfp_field* field);
        };
        
        std::unique_ptr<zfp_stream, ZFPStreamDeleter> stream_;
        std::unique_ptr<zfp_field, ZFPFieldDeleter> field_;
        
        bool initialize_stream();
        size_t compress_internal(void* buffer, size_t bufsize);
        size_t decompress_internal(const void* src, size_t src_size);
    };
#else
    // Fallback when ZFP is not available
    class ZFPCompression : public CompressionBackend {
    public:
        explicit ZFPCompression(const ZFPParams& params = ZFPParams()) : params_(params) {}
        bool compress(const void* src, size_t src_size, std::vector<uint8_t>& out) override { return false; }
        bool decompress(const uint8_t* src_data, size_t src_size, size_t decompressed_size, void* dst) override { return false; }
        static bool initialize() { return false; }
        static void shutdown() {}
        double get_last_compression_ratio() const { return 0.0; }
        void set_params(const ZFPParams& params) { params_ = params; }
        const ZFPParams& get_params() const { return params_; }
    private:
        ZFPParams params_;
    };
#endif

} // namespace compression
