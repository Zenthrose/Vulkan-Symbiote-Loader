#include "Common.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace vk_symbiote {

class ZFPDecompressor {
public:
    ZFPDecompressor(uint32_t minbits = 4, uint32_t maxbits = 64, uint32_t maxprec = 32)
        : minbits_(minbits), maxbits_(maxbits), maxprec_(maxprec) {}

    Expected<std::vector<float>> decompress(
        const uint8_t* compressed_data,
        size_t compressed_size,
        uint64_t original_size,
        uint32_t /*dims*/ = 1) {

        std::vector<float> result(original_size, 0.0f);
        
        const uint8_t* ptr = compressed_data;
        uint64_t remaining = original_size;
        
        while (remaining > 0) {
            uint32_t block_size = std::min(static_cast<uint32_t>(remaining), 64u);
            
            if (!decompress_block(ptr, compressed_size - (ptr - compressed_data), 
                                   result.data() + (original_size - remaining), block_size)) {
                return Expected<std::vector<float>>(-1);
            }
            
            ptr += estimate_compressed_size(block_size);
            remaining -= block_size;
        }
        
        return Expected<std::vector<float>>(std::move(result));
    }

private:
    uint32_t minbits_;
    uint32_t maxbits_;
    uint32_t maxprec_;

    bool decompress_block(const uint8_t* data, size_t size, float* output, uint32_t count) {
        if (count == 0 || count > 64) return false;
        
        const uint8_t* ptr = data;
        
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t bits = minbits_;
            uint32_t mantissa = 0;
            
            for (uint32_t b = 0; b < 4 && ptr < data + size; ++b) {
                mantissa |= (*ptr++ << (b * 8));
                bits = std::min(bits + 8, maxbits_);
            }
            
            float decoded = decode_float(mantissa, bits);
            output[i] = decoded;
        }
        
        return true;
    }

    float decode_float(uint32_t mantissa, uint32_t bits) {
        uint32_t value;
        if (bits <= 23) {
            value = mantissa << (23 - bits);
        } else {
            uint32_t exp = bits - 23;
            value = mantissa >> exp;
        }
        float result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    size_t estimate_compressed_size(uint32_t count) const {
        return count * (maxbits_ / 8 + 1);
    }
};

class BloscDecompressor {
public:
    Expected<std::vector<float>> decompress(
        const uint8_t* compressed_data,
        size_t compressed_size,
        uint64_t original_size) {

        std::vector<float> result(original_size, 0.0f);
        
        const uint8_t* header = compressed_data;
        (void)*reinterpret_cast<const uint32_t*>(header);  // nbytes
        uint32_t cbytes = *reinterpret_cast<const uint32_t*>(header + 4);
        (void)*reinterpret_cast<const uint32_t*>(header + 8);  // blocksize
        uint8_t codec = header[12];
        (void)header[13];  // filters
        (void)header[14];  // typesize
        
        const uint8_t* data_ptr = compressed_data + 16;
        
        if (codec == 0) {
            std::memcpy(result.data(), data_ptr, std::min(compressed_size - 16, original_size * sizeof(float)));
        } else if (codec == 2) {
            if (!decompress_lz4(data_ptr, cbytes, reinterpret_cast<uint8_t*>(result.data()), original_size * sizeof(float))) {
                return Expected<std::vector<float>>(-1);
            }
        }
        
        return Expected<std::vector<float>>(std::move(result));
    }

private:
    bool decompress_lz4(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size) {
        const uint8_t* ip = src;
        const uint8_t* const iend = ip + src_size;
        uint8_t* op = dst;
        uint8_t* const oend = op + dst_size;

        while (ip < iend) {
            uint8_t token = *ip++;
            uint32_t lit_len = token >> 4;
            
            if (lit_len == 15) {
                uint8_t len = *ip++;
                while (len == 255) {
                    len = *ip++;
                    lit_len += len;
                }
            }
            
            if (ip + lit_len > iend) return false;
            std::memcpy(op, ip, lit_len);
            ip += lit_len;
            op += lit_len;
            
            if (ip >= iend) break;
            
            uint32_t match = *reinterpret_cast<const uint16_t*>(ip);
            ip += 2;
            uint32_t match_len = token & 0xF;
            
            if (match_len == 15) {
                uint8_t len = *ip++;
                while (len == 255) {
                    len = *ip++;
                    match_len += len;
                }
            }
            match_len += 4;
            
            const uint8_t* ref = op - match;
            std::memcpy(op, ref, match_len);
            op += match_len;
        }
        
        return op == oend;
    }
};

class Compression {
public:
    enum class Codec : uint8_t {
        NONE = 0,
        ZFP = 1,
        BLOSC = 2,
        LZ4 = 3,
        ZSTD = 4
    };

    static Expected<std::vector<float>> decompress(
        const uint8_t* compressed_data,
        size_t compressed_size,
        uint64_t original_size,
        Codec codec = Codec::BLOSC) {

        switch (codec) {
            case Codec::ZFP: {
                ZFPDecompressor zfp;
                return zfp.decompress(compressed_data, compressed_size, original_size);
            }
            case Codec::BLOSC:
            case Codec::LZ4: {
                BloscDecompressor blosc;
                return blosc.decompress(compressed_data, compressed_size, original_size);
            }
            case Codec::NONE:
            default: {
                std::vector<float> result(original_size);
                size_t copy_size = std::min(compressed_size, original_size * sizeof(float));
                std::memcpy(result.data(), compressed_data, copy_size);
                return Expected<std::vector<float>>(std::move(result));
            }
        }
    }

    static Expected<std::vector<uint8_t>> compress(
        const float* data,
        size_t size,
        Codec codec = Codec::BLOSC,
        float tolerance = 0.001f) {

        std::vector<uint8_t> result;
        
        switch (codec) {
            case Codec::ZFP:
                result = compress_zfp(data, size, tolerance);
                break;
            case Codec::BLOSC:
                result = compress_blosc(data, size);
                break;
            default:
                result.resize(size * sizeof(float));
                std::memcpy(result.data(), data, size * sizeof(float));
                break;
        }
        
        return Expected<std::vector<uint8_t>>(std::move(result));
    }

private:
    static std::vector<uint8_t> compress_zfp(const float* data, size_t size, float tolerance) {
        std::vector<uint8_t> result;
        
        const uint32_t prec = static_cast<uint32_t>(-std::log2(tolerance));
        
        size_t offset = 0;
        while (offset < size) {
            uint32_t block_size = static_cast<uint32_t>(std::min(size - offset, static_cast<size_t>(64)));
            
            for (uint32_t i = 0; i < block_size; ++i) {
                uint32_t encoded = encode_float(data[offset + i], prec);
                size_t byte_offset = result.size();
                result.resize(byte_offset + 4);
                std::memcpy(result.data() + byte_offset, &encoded, 4);
            }
            
            offset += block_size;
        }
        
        return result;
    }

    static uint32_t encode_float(float value, uint32_t prec) {
        uint32_t result;
        std::memcpy(&result, &value, 4);
        
        uint32_t mantissa = result & 0x7FFFFF;
        mantissa = mantissa >> (23 - prec);
        
        result = (result & 0xFF800000) | mantissa;
        return result;
    }

    static std::vector<uint8_t> compress_blosc(const float* data, size_t size) {
        std::vector<uint8_t> result;
        result.resize(size * sizeof(float) + 16);
        
        *reinterpret_cast<uint32_t*>(result.data()) = size * sizeof(float);
        *reinterpret_cast<uint32_t*>(result.data() + 4) = size * sizeof(float);
        *reinterpret_cast<uint32_t*>(result.data() + 8) = 0;
        result[12] = 0;
        result[13] = 0;
        result[14] = 4;
        result[15] = 0;
        
        std::memcpy(result.data() + 16, data, size * sizeof(float));
        
        return result;
    }
};

} // namespace vk_symbiote
