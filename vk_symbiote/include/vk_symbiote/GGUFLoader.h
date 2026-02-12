#pragma once

#include "Common.h"
#include "NomadPack.h"
#include <fstream>
#include <cstring>
#include <algorithm>

namespace vk_symbiote {

enum class GGUFValueType : uint32 {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    Q4_0 = 13,
    Q4_1 = 14,
    Q5_0 = 15,
    Q5_1 = 16,
    Q8_0 = 17,
    Q8_1 = 18,
    Q2_K = 19,
    Q3_K = 20,
    Q4_K = 21,
    Q5_K = 22,
    Q6_K = 23,
    Q8_K = 24,
    IQ1_S = 25,
    IQ1_M = 26,
    IQ2_XXS = 27,
    IQ2_XS = 28,
    IQ3_XXS = 29,
    IQ4_XS = 30,
    IQ4_NL = 31,
    IQ5_S = 32,
    IQ5_M = 33,
    IQ6_N = 34,
    IQ6_1 = 35,
    IQ2_S = 36,
    IQ3_S = 37,
    IQ4_1 = 38,
    IQ3_1 = 39,
    IQ2_1 = 40,
    IQ1_1 = 41,
    IQ4_NL_2 = 42,
    IQ3_L = 43,
    IQ3_XS = 44,
    IQ3_M = 45,
    IQ4_XS_2 = 46,
    BFLOAT16 = 47,
    FLOAT16 = 48,
    IQ2_M = 49,
    IQ3_XS_2 = 50,
    IQ3_L_2 = 51,
    IQ3_M_2 = 52,
    IQ4_XS_3 = 53
};

struct GGUFHeader {
    uint32 magic = 0;
    uint32 version = 0;
    uint64 tensor_count = 0;
    uint64 metadata_kv_count = 0;
};

struct GGUFTensorInfo {
    std::string name;
    uint32 n_dimensions = 0;
    std::vector<uint64> dimensions;
    GGUFValueType data_type;
    uint64 offset;
};

struct GGUFMetadataKV {
    std::string key;
    GGUFValueType value_type;
    void* value = nullptr;
};

class GGUFLoader {
public:
    explicit GGUFLoader(const Path& file_path);
    ~GGUFLoader();

    ExpectedVoid load();
    ExpectedVoid close();

    ModelConfig get_model_config() const;

    std::vector<PackMetadata> generate_packs(const ModelConfig& config);

    uint64 tensor_count() const;
    uint64 metadata_count() const;

    const GGUFTensorInfo* get_tensor(const std::string& name) const;
    Expected<std::vector<float>> read_tensor_data(const GGUFTensorInfo& tensor, bool fp16_to_fp32 = true);

    static uint64 align_offset(uint64 offset, uint64 alignment);

private:
    Path file_path_;
    class GGUFLoaderImpl;
    std::unique_ptr<GGUFLoaderImpl> pimpl_;
};

} // namespace vk_symbiote
