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
    GGUFLoader(const Path& file_path);
    ~GGUFLoader();

    ExpectedVoid load();
    ExpectedVoid close();

    ModelConfig get_model_config() const { return model_config_; }

    std::vector<PackMetadata> generate_packs(const ModelConfig& config);

    uint64 tensor_count() const { return tensors_.size(); }
    uint64 metadata_count() const { return metadata_.size(); }

    const GGUFTensorInfo* get_tensor(const std::string& name) const;
    Expected<std::vector<float>> read_tensor_data(const GGUFTensorInfo& tensor, bool fp16_to_fp32 = true);

    static uint64 align_offset(uint64 offset, uint64 alignment);

private:
    Path file_path_;
    std::ifstream file_;
    GGUFHeader header_;
    std::vector<GGUFMetadataKV> metadata_;
    std::vector<GGUFTensorInfo> tensors_;
    ModelConfig model_config_;
    uint64 data_offset_ = 0;

    ExpectedVoid read_header();
    ExpectedVoid read_metadata();
    ExpectedVoid read_tensors();

    void parse_model_type();
    void parse_hyperparameters();

    ExpectedVoid read_string(std::string& str);
    ExpectedVoid read_value(void* value, GGUFValueType type);

    template<typename T>
    ExpectedVoid read_array(std::vector<T>& arr, GGUFValueType element_type);
};

inline GGUFLoader::GGUFLoader(const Path& file_path) : file_path_(file_path) {
}

inline GGUFLoader::~GGUFLoader() {
    close();
}

inline ExpectedVoid GGUFLoader::load() {
    file_.open(file_path_.c_str(), std::ios::binary);
    if (!file_.is_open()) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }

    auto result = read_header();
    if (!result.has_value()) {
        return result;
    }

    result = read_metadata();
    if (!result.has_value()) {
        return result;
    }

    result = read_tensors();
    if (!result.has_value()) {
        return result;
    }

    parse_model_type();
    parse_hyperparameters();

    return make_expected_success();
}

inline ExpectedVoid GGUFLoader::close() {
    if (file_.is_open()) {
        file_.close();
    }

    for (auto& kv : metadata_) {
        delete[] static_cast<char*>(kv.value);
        kv.value = nullptr;
    }
    metadata_.clear();
    tensors_.clear();

    return make_expected_success();
}

inline ExpectedVoid GGUFLoader::read_header() {
    char magic[4];
    file_.read(magic, 4);

    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }

    file_.read(reinterpret_cast<char*>(&header_.version), 4);
    file_.read(reinterpret_cast<char*>(&header_.tensor_count), 8);
    file_.read(reinterpret_cast<char*>(&header_.metadata_kv_count), 8);

    return make_expected_success();
}

inline ExpectedVoid GGUFLoader::read_metadata() {
    for (uint64 i = 0; i < header_.metadata_kv_count; ++i) {
        GGUFMetadataKV kv;

        auto result = read_string(kv.key);
        if (!result.has_value()) {
            return result;
        }

        file_.read(reinterpret_cast<char*>(&kv.value_type), 4);

        switch (kv.value_type) {
            case GGUFValueType::UINT8: {
                uint8* val = new uint8();
                file_.read(reinterpret_cast<char*>(val), 1);
                kv.value = val;
                break;
            }
            case GGUFValueType::INT8: {
                int8* val = new int8();
                file_.read(reinterpret_cast<char*>(val), 1);
                kv.value = val;
                break;
            }
            case GGUFValueType::UINT32: {
                uint32* val = new uint32();
                file_.read(reinterpret_cast<char*>(val), 4);
                kv.value = val;
                break;
            }
            case GGUFValueType::INT32: {
                int32* val = new int32();
                file_.read(reinterpret_cast<char*>(val), 4);
                kv.value = val;
                break;
            }
            case GGUFValueType::FLOAT32: {
                float* val = new float();
                file_.read(reinterpret_cast<char*>(val), 4);
                kv.value = val;
                break;
            }
            case GGUFValueType::STRING: {
                std::string* val = new std::string();
                auto result = read_string(*val);
                if (!result.has_value()) {
                    delete val;
                    return result;
                }
                kv.value = val;
                break;
            }
            case GGUFValueType::UINT64: {
                uint64* val = new uint64();
                file_.read(reinterpret_cast<char*>(val), 8);
                kv.value = val;
                break;
            }
            case GGUFValueType::INT64: {
                int64* val = new int64();
                file_.read(reinterpret_cast<char*>(val), 8);
                kv.value = val;
                break;
            }
            default:
                kv.value = nullptr;
                break;
        }

        metadata_.push_back(kv);
    }

    return make_expected_success();
}

inline ExpectedVoid GGUFLoader::read_tensors() {
    uint64 current_offset = data_offset_;

    for (uint64 i = 0; i < header_.tensor_count; ++i) {
        GGUFTensorInfo tensor;

        auto result = read_string(tensor.name);
        if (!result.has_value()) {
            return result;
        }

        file_.read(reinterpret_cast<char*>(&tensor.n_dimensions), 4);
        tensor.dimensions.resize(tensor.n_dimensions);

        for (uint32 j = 0; j < tensor.n_dimensions; ++j) {
            file_.read(reinterpret_cast<char*>(&tensor.dimensions[j]), 8);
        }

        file_.read(reinterpret_cast<char*>(&tensor.data_type), 4);
        file_.read(reinterpret_cast<char*>(&tensor.offset), 8);

        uint64 tensor_size = 1;
        for (uint64 dim : tensor.dimensions) {
            tensor_size *= dim;
        }

        uint64 type_size = 2;
        if (tensor.data_type == GGUFValueType::FLOAT32 ||
            tensor.data_type == GGUFValueType::UINT32 ||
            tensor.data_type == GGUFValueType::INT32) {
            type_size = 4;
        } else if (tensor.data_type == GGUFValueType::FLOAT64 ||
                   tensor.data_type == GGUFValueType::UINT64 ||
                   tensor.data_type == GGUFValueType::INT64) {
            type_size = 8;
        }

        tensor.offset = align_offset(tensor.offset, 32);
        current_offset = std::max(current_offset, tensor.offset + tensor_size * type_size);

        tensors_.push_back(tensor);
    }

    data_offset_ = align_offset(current_offset, 4096);

    return make_expected_success();
}

inline void GGUFLoader::parse_model_type() {
    for (const auto& kv : metadata_) {
        if (kv.key == "general.architecture") {
            if (kv.value_type == GGUFValueType::STRING) {
                std::string* arch = reinterpret_cast<std::string*>(kv.value);
                model_config_.model_type = *arch;
            }
        }
    }
}

inline void GGUFLoader::parse_hyperparameters() {
    for (const auto& kv : metadata_) {
        if (kv.key == "llama.context_length" || kv.key == "llama.contextLength") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.max_position_embeddings = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.embedding_length" || kv.key == "llama.embeddingLength") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.hidden_size = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.feed_forward_length" || kv.key == "llama.feedForwardLength") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.intermediate_size = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.attention.head_count" || kv.key == "llama.attention.headCount") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.num_attention_heads = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.attention.head_count_kv" || kv.key == "llama.attention.headCountKV") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.num_key_value_heads = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.block_count" || kv.key == "llama.blockCount") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.num_layers = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.vocab_size" || kv.key == "llama.vocabSize") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.vocab_size = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.rope.dimension_count" || kv.key == "llama.rope.dimensionCount") {
            if (kv.value_type == GGUFValueType::UINT32) {
                model_config_.head_dim = *reinterpret_cast<uint32*>(kv.value);
            }
        } else if (kv.key == "llama.rope.freq_base" || kv.key == "llama.rope.freqBase") {
            if (kv.value_type == GGUFValueType::FLOAT32) {
                model_config_.rope_theta = *reinterpret_cast<float*>(kv.value);
            }
        } else if (kv.key == "llama.attention.layer_norm_rms_epsilon" || kv.key == "llama.attention.layerNormEpsilon") {
            if (kv.value_type == GGUFValueType::FLOAT32) {
                model_config_.rms_epsilon = *reinterpret_cast<float*>(kv.value);
            }
        }
    }

    model_config_.head_dim = model_config_.hidden_size / model_config_.num_attention_heads;
}

inline ExpectedVoid GGUFLoader::read_string(std::string& str) {
    uint64 len;
    file_.read(reinterpret_cast<char*>(&len), 8);

    str.resize(len);
    file_.read(&str[0], static_cast<std::streamsize>(len));

    return make_expected_success();
}

inline ExpectedVoid GGUFLoader::read_value(void* value, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:
            file_.read(static_cast<char*>(value), 1);
            break;
        case GGUFValueType::INT8:
            file_.read(static_cast<char*>(value), 1);
            break;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            file_.read(static_cast<char*>(value), 4);
            break;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            file_.read(static_cast<char*>(value), 8);
            break;
        default:
            break;
    }
    return make_expected_success();
}

template<typename T>
inline ExpectedVoid GGUFLoader::read_array(std::vector<T>& arr, GGUFValueType element_type) {
    uint64 len;
    file_.read(reinterpret_cast<char*>(&len), 8);

    arr.resize(len);
    for (uint64 i = 0; i < len; ++i) {
        auto result = read_value(&arr[i], element_type);
        if (!result.has_value()) {
            return result;
        }
    }
    return make_expected_success();
}

inline const GGUFTensorInfo* GGUFLoader::get_tensor(const std::string& name) const {
    for (const auto& tensor : tensors_) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

inline Expected<std::vector<float>> GGUFLoader::read_tensor_data(const GGUFTensorInfo& tensor, bool fp16_to_fp32) {
    file_.seekg(static_cast<std::streamoff>(tensor.offset), std::ios::beg);

    uint64 element_count = 1;
    for (uint64 dim : tensor.dimensions) {
        element_count *= dim;
    }

    std::vector<float> result(element_count);

    if (tensor.data_type == GGUFValueType::FLOAT32) {
        file_.read(reinterpret_cast<char*>(result.data()), element_count * sizeof(float));
    } else if (tensor.data_type == GGUFValueType::FLOAT16) {
        std::vector<uint16_t> buffer(element_count);
        file_.read(reinterpret_cast<char*>(buffer.data()), element_count * sizeof(uint16_t));

        for (uint64 i = 0; i < element_count; ++i) {
            uint16_t val = buffer[i];
            int sign = (val >> 15) & 1;
            int exp = (val >> 10) & 0x1F;
            int mantissa = val & 0x3FF;

            if (exp == 0) {
                result[i] = mantissa * std::pow(2, -14);
            } else if (exp == 31) {
                result[i] = (sign ? -1.0f : 1.0f) * std::numeric_limits<float>::infinity();
            } else {
                float normalized = 1.0f + mantissa / 1024.0f;
                result[i] = (sign ? -1.0f : 1.0f) * normalized * std::pow(2, exp - 15);
            }
        }
    } else if (tensor.data_type == GGUFValueType::BFLOAT16) {
        std::vector<uint16_t> buffer(element_count);
        file_.read(reinterpret_cast<char*>(buffer.data()), element_count * sizeof(uint16_t));

        for (uint64 i = 0; i < element_count; ++i) {
            uint16_t bval = buffer[i];
            uint32_t fval = static_cast<uint32_t>(bval) << 16;
            std::memcpy(&result[i], &fval, sizeof(float));
        }
    }

    return Expected<std::vector<float>>(std::move(result));
}

inline uint64 GGUFLoader::align_offset(uint64 offset, uint64 alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) / alignment * alignment;
}

inline std::vector<PackMetadata> GGUFLoader::generate_packs(const ModelConfig& config) {
    std::vector<PackMetadata> packs;

    uint64 pack_id = 0;

    for (uint32 layer = 0; layer < config.num_layers; ++layer) {
        std::string layer_prefix = "layers." + std::to_string(layer) + ".";

        std::string attention_norm_name = layer_prefix + "attention_norm.weight";
        if (get_tensor(attention_norm_name)) {
            PackMetadata pack;
            pack.pack_id = pack_id++;
            pack.type = PackType::NORM_GAMMA;
            pack.layer_idx = layer;
            pack.tensor_name = attention_norm_name;
            pack.base_priority = 0.9f - static_cast<float>(layer) / config.num_layers * 0.1f;
            packs.push_back(pack);
        }

        for (uint32 head = 0; head < config.num_attention_heads; ++head) {
            std::string q_name = layer_prefix + "attention.wq.weight";
            std::string k_name = layer_prefix + "attention.wk.weight";
            std::string v_name = layer_prefix + "attention.wv.weight";

            if (get_tensor(q_name)) {
                PackMetadata q_pack;
                q_pack.pack_id = pack_id++;
                q_pack.type = PackType::ATTENTION_Q;
                q_pack.layer_idx = layer;
                q_pack.head_idx = head;
                q_pack.head_group_idx = head;
                q_pack.tensor_name = q_name;
                q_pack.base_priority = 0.8f - static_cast<float>(layer) / config.num_layers * 0.1f;
                packs.push_back(q_pack);
            }

            if (get_tensor(k_name)) {
                PackMetadata k_pack;
                k_pack.pack_id = pack_id++;
                k_pack.type = PackType::ATTENTION_K;
                k_pack.layer_idx = layer;
                k_pack.head_idx = head;
                k_pack.head_group_idx = head;
                k_pack.tensor_name = k_name;
                k_pack.base_priority = 0.7f - static_cast<float>(layer) / config.num_layers * 0.1f;
                packs.push_back(k_pack);
            }

            if (get_tensor(v_name)) {
                PackMetadata v_pack;
                v_pack.pack_id = pack_id++;
                v_pack.type = PackType::ATTENTION_V;
                v_pack.layer_idx = layer;
                v_pack.head_idx = head;
                v_pack.head_group_idx = head;
                v_pack.tensor_name = v_name;
                v_pack.base_priority = 0.7f - static_cast<float>(layer) / config.num_layers * 0.1f;
                packs.push_back(v_pack);
            }
        }

        std::string o_name = layer_prefix + "attention.wo.weight";
        if (get_tensor(o_name)) {
            PackMetadata pack;
            pack.pack_id = pack_id++;
            pack.type = PackType::ATTENTION_O;
            pack.layer_idx = layer;
            pack.tensor_name = o_name;
            pack.base_priority = 0.7f - static_cast<float>(layer) / config.num_layers * 0.1f;
            packs.push_back(pack);
        }

        std::string feed_forward_up_name = layer_prefix + "feed_forward.w1.weight";
        std::string feed_forward_gate_name = layer_prefix + "feed_forward.w3.weight";
        std::string feed_forward_down_name = layer_prefix + "feed_forward.w2.weight";

        if (get_tensor(feed_forward_up_name)) {
            PackMetadata pack;
            pack.pack_id = pack_id++;
            pack.type = PackType::FEED_FORWARD_UP;
            pack.layer_idx = layer;
            pack.tensor_name = feed_forward_up_name;
            pack.base_priority = 0.6f - static_cast<float>(layer) / config.num_layers * 0.1f;
            packs.push_back(pack);
        }

        if (get_tensor(feed_forward_gate_name)) {
            PackMetadata pack;
            pack.pack_id = pack_id++;
            pack.type = PackType::FEED_FORWARD_GATE;
            pack.layer_idx = layer;
            pack.tensor_name = feed_forward_gate_name;
            pack.base_priority = 0.6f - static_cast<float>(layer) / config.num_layers * 0.1f;
            packs.push_back(pack);
        }

        if (get_tensor(feed_forward_down_name)) {
            PackMetadata pack;
            pack.pack_id = pack_id++;
            pack.type = PackType::FEED_FORWARD_DOWN;
            pack.layer_idx = layer;
            pack.tensor_name = feed_forward_down_name;
            pack.base_priority = 0.6f - static_cast<float>(layer) / config.num_layers * 0.1f;
            packs.push_back(pack);
        }
    }

    return packs;
}

} // namespace vk_symbiote
