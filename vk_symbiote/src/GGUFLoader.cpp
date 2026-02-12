#include "GGUFLoader.h"
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <thread>
#include <future>
#include <unordered_map>

namespace vk_symbiote {

// Static tensor type size lookup table
static uint64_t get_type_size(GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::INT8:
            return 1;
        case GGUFValueType::UINT16:
        case GGUFValueType::INT16:
        case GGUFValueType::FLOAT16:
        case GGUFValueType::BFLOAT16:
            return 2;
        case GGUFValueType::UINT32:
        case GGUFValueType::INT32:
        case GGUFValueType::FLOAT32:
            return 4;
        case GGUFValueType::UINT64:
        case GGUFValueType::INT64:
        case GGUFValueType::FLOAT64:
            return 8;
        case GGUFValueType::Q4_0:
        case GGUFValueType::Q4_1:
            return 18; // 4 bits per element, 32 elements per block + metadata
        case GGUFValueType::Q5_0:
        case GGUFValueType::Q5_1:
            return 22; // 5 bits per element
        case GGUFValueType::Q8_0:
        case GGUFValueType::Q8_1:
            return 34; // 8 bits per element
        default:
            return 4; // Default to float32
    }
}

// Calculate total tensor size in bytes
static uint64_t calculate_tensor_size(const GGUFTensorInfo& tensor) {
    uint64_t element_count = 1;
    for (uint64_t dim : tensor.dimensions) {
        element_count *= dim;
    }
    
    uint64_t type_size = get_type_size(tensor.data_type);
    
    // Handle quantized types with block structure
    if (tensor.data_type >= GGUFValueType::Q4_0 && tensor.data_type <= GGUFValueType::Q8_1) {
        const uint64_t block_size = 32;
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        return num_blocks * type_size;
    }
    
    return element_count * type_size;
}

// Streaming tensor reader for on-demand loading
class StreamingTensorReader {
public:
    explicit StreamingTensorReader(const std::filesystem::path& path) 
        : file_path_(path), file_stream_() {}
    
    ~StreamingTensorReader() {
        close();
    }
    
    bool open() {
        file_stream_.open(file_path_, std::ios::binary | std::ios::in);
        return file_stream_.is_open();
    }
    
    void close() {
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
    }
    
    // Read a chunk of tensor data at specified offset
    bool read_chunk(uint64_t offset, void* buffer, size_t size) {
        if (!file_stream_.is_open()) return false;
        
        file_stream_.seekg(static_cast<std::streamoff>(offset));
        if (!file_stream_.good()) return false;
        
        file_stream_.read(static_cast<char*>(buffer), static_cast<std::streamsize>(size));
        return file_stream_.good() || file_stream_.gcount() == static_cast<std::streamsize>(size);
    }
    
    // Async read using future/promise pattern
    std::future<bool> read_chunk_async(uint64_t offset, void* buffer, size_t size) {
        return std::async(std::launch::async, [this, offset, buffer, size]() {
            return read_chunk(offset, buffer, size);
        });
    }
    
private:
    std::filesystem::path file_path_;
    std::ifstream file_stream_;
};

// Tensor pack mapping - maps GGUF tensors to NomadPack structures
class TensorPackMapper {
public:
    struct PackMapping {
        uint64_t pack_id;
        std::string tensor_name;
        uint64_t file_offset;
        uint64_t compressed_size;
        uint64_t decompressed_size;
        GGUFValueType data_type;
        std::vector<uint64_t> dimensions;
        uint32_t layer_idx;
        PackType pack_type;
    };
    
    std::vector<PackMapping> create_mappings(
        const std::vector<GGUFTensorInfo>& tensors,
        const ModelConfig& config) {
        
        std::vector<PackMapping> mappings;
        uint64_t next_pack_id = 0;
        
        // Group tensors by layer for efficient access patterns
        std::unordered_map<uint32_t, std::vector<const GGUFTensorInfo*>> layer_tensors;
        
        for (const auto& tensor : tensors) {
            uint32_t layer_idx = extract_layer_index(tensor.name);
            layer_tensors[layer_idx].push_back(&tensor);
        }
        
        // Create pack mappings for each layer
        for (const auto& [layer_idx, tensor_ptrs] : layer_tensors) {
            for (const auto* tensor : tensor_ptrs) {
                PackMapping mapping;
                mapping.pack_id = next_pack_id++;
                mapping.tensor_name = tensor->name;
                mapping.file_offset = tensor->offset;
                mapping.data_type = tensor->data_type;
                mapping.dimensions = tensor->dimensions;
                mapping.layer_idx = layer_idx;
                mapping.pack_type = infer_pack_type(tensor->name);
                
                // Calculate sizes
                mapping.decompressed_size = calculate_tensor_size(*tensor);
                mapping.compressed_size = mapping.decompressed_size; // GGUF stores raw, we'll compress in NomadPack
                
                mappings.push_back(mapping);
            }
        }
        
        return mappings;
    }
    
private:
    uint32_t extract_layer_index(const std::string& tensor_name) {
        // Parse layer index from tensor name (e.g., "layers.5.attention.wq.weight" -> 5)
        size_t pos = tensor_name.find("layers.");
        if (pos != std::string::npos) {
            size_t start = pos + 7; // length of "layers."
            size_t end = tensor_name.find('.', start);
            if (end != std::string::npos) {
                try {
                    return std::stoul(tensor_name.substr(start, end - start));
                } catch (...) {
                    return 0;
                }
            }
        }
        return 0;
    }
    
    PackType infer_pack_type(const std::string& tensor_name) {
        if (tensor_name.find("attention.wq") != std::string::npos) return PackType::ATTENTION_Q;
        if (tensor_name.find("attention.wk") != std::string::npos) return PackType::ATTENTION_K;
        if (tensor_name.find("attention.wv") != std::string::npos) return PackType::ATTENTION_V;
        if (tensor_name.find("attention.wo") != std::string::npos) return PackType::ATTENTION_O;
        if (tensor_name.find("feed_forward.w1") != std::string::npos) return PackType::FEED_FORWARD_UP;
        if (tensor_name.find("feed_forward.w2") != std::string::npos) return PackType::FEED_FORWARD_DOWN;
        if (tensor_name.find("feed_forward.w3") != std::string::npos) return PackType::FEED_FORWARD_GATE;
        if (tensor_name.find("attention_norm") != std::string::npos) return PackType::NORM_GAMMA;
        if (tensor_name.find("ffn_norm") != std::string::npos) return PackType::NORM_GAMMA;
        if (tensor_name.find("rope") != std::string::npos) return PackType::ROPE;
        if (tensor_name.find("embed") != std::string::npos || tensor_name.find("token_embd") != std::string::npos) 
            return PackType::EMBEDDING;
        if (tensor_name.find("output") != std::string::npos || tensor_name.find("output_norm") != std::string::npos)
            return PackType::HEAD;
        return PackType::UNKNOWN;
    }
};

// Full GGUFLoader implementation with streaming support
class GGUFLoaderImpl {
public:
    explicit GGUFLoaderImpl(const Path& file_path) 
        : file_path_(file_path), data_offset_(0) {}
    
    ExpectedVoid load() {
        // Open file
        file_.open(file_path_.c_str(), std::ios::binary);
        if (!file_.is_open()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        // Read header
        auto header_result = read_header();
        if (!header_result.has_value()) {
            return header_result;
        }
        
        // Read metadata
        auto metadata_result = read_metadata();
        if (!metadata_result.has_value()) {
            return metadata_result;
        }
        
        // Read tensor info
        auto tensor_result = read_tensors();
        if (!tensor_result.has_value()) {
            return tensor_result;
        }
        
        // Parse model configuration from metadata
        parse_model_config();
        
        // Create tensor pack mappings
        TensorPackMapper mapper;
        tensor_mappings_ = mapper.create_mappings(tensors_, model_config_);
        
        // Initialize streaming reader
        stream_reader_ = std::make_unique<StreamingTensorReader>(file_path_);
        if (!stream_reader_->open()) {
            return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        return make_expected_success();
    }
    
    ExpectedVoid close() {
        if (file_.is_open()) {
            file_.close();
        }
        if (stream_reader_) {
            stream_reader_->close();
        }
        
        // Clean up metadata values
        for (auto& kv : metadata_) {
            if (kv.value) {
                free_metadata_value(kv.value_type, kv.value);
                kv.value = nullptr;
            }
        }
        
        return make_expected_success();
    }
    
    // On-demand tensor data loading - only reads requested tensor from disk
    Expected<std::vector<float>> load_tensor_data(const std::string& tensor_name, bool convert_fp16 = true) {
        const GGUFTensorInfo* tensor = find_tensor(tensor_name);
        if (!tensor) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
        
        uint64_t tensor_size = calculate_tensor_size(*tensor);
        uint64_t element_count = tensor_size / get_type_size(tensor->data_type);
        
        std::vector<float> result;
        result.reserve(element_count);
        
        // Read data based on type
        switch (tensor->data_type) {
            case GGUFValueType::FLOAT32: {
                std::vector<float> buffer(element_count);
                if (!stream_reader_->read_chunk(tensor->offset, buffer.data(), tensor_size)) {
                    return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
                }
                result = std::move(buffer);
                break;
            }
            
            case GGUFValueType::FLOAT16: {
                if (convert_fp16) {
                    std::vector<uint16_t> buffer(element_count);
                    if (!stream_reader_->read_chunk(tensor->offset, buffer.data(), tensor_size)) {
                        return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
                    }
                    result.reserve(element_count);
                    for (uint16_t val : buffer) {
                        result.push_back(fp16_to_fp32(val));
                    }
                } else {
                    // Return as float but stored as uint16_t bits
                    std::vector<uint16_t> buffer(element_count);
                    if (!stream_reader_->read_chunk(tensor->offset, buffer.data(), tensor_size)) {
                        return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
                    }
                    result.resize(element_count);
                    std::memcpy(result.data(), buffer.data(), tensor_size);
                }
                break;
            }
            
            case GGUFValueType::BFLOAT16: {
                std::vector<uint16_t> buffer(element_count);
                if (!stream_reader_->read_chunk(tensor->offset, buffer.data(), tensor_size)) {
                    return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
                }
                result.reserve(element_count);
                for (uint16_t val : buffer) {
                    result.push_back(bf16_to_fp32(val));
                }
                break;
            }
            
            case GGUFValueType::Q8_0: {
                result = decompress_q8_0(*tensor);
                break;
            }
            
            case GGUFValueType::Q4_0: {
                result = decompress_q4_0(*tensor);
                break;
            }
            
            default:
                // For other types, read as raw bytes and convert
                std::vector<uint8_t> buffer(tensor_size);
                if (!stream_reader_->read_chunk(tensor->offset, buffer.data(), tensor_size)) {
                    return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
                }
                result = convert_to_fp32(buffer, tensor->data_type, element_count);
                break;
        }
        
        return Expected<std::vector<float>>(std::move(result));
    }
    
    // Generate NomadPack metadata from tensor mappings
    std::vector<PackMetadata> generate_packs() {
        std::vector<PackMetadata> packs;
        packs.reserve(tensor_mappings_.size());
        
        for (const auto& mapping : tensor_mappings_) {
            PackMetadata pack;
            pack.pack_id = mapping.pack_id;
            pack.type = mapping.pack_type;
            pack.layer_idx = mapping.layer_idx;
            pack.file_offset = mapping.file_offset;
            pack.compressed_size = mapping.compressed_size;
            pack.decompressed_size = mapping.decompressed_size;
            pack.tensor_name = mapping.tensor_name;
            
            // Set base priority based on layer and type
            float layer_factor = 1.0f - (static_cast<float>(mapping.layer_idx) / model_config_.num_layers);
            float type_factor = 0.8f;
            
            switch (mapping.pack_type) {
                case PackType::ATTENTION_Q:
                case PackType::ATTENTION_K:
                case PackType::ATTENTION_V:
                    type_factor = 1.0f;
                    break;
                case PackType::ATTENTION_O:
                    type_factor = 0.9f;
                    break;
                case PackType::FEED_FORWARD_UP:
                case PackType::FEED_FORWARD_GATE:
                case PackType::FEED_FORWARD_DOWN:
                    type_factor = 0.7f;
                    break;
                default:
                    type_factor = 0.5f;
                    break;
            }
            
            pack.base_priority = layer_factor * type_factor;
            packs.push_back(pack);
        }
        
        return packs;
    }
    
    const GGUFHeader& header() const { return header_; }
    const ModelConfig& model_config() const { return model_config_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    const std::vector<GGUFMetadataKV>& metadata() const { return metadata_; }
    
private:
    Path file_path_;
    std::ifstream file_;
    std::unique_ptr<StreamingTensorReader> stream_reader_;
    
    GGUFHeader header_;
    std::vector<GGUFMetadataKV> metadata_;
    std::vector<GGUFTensorInfo> tensors_;
    ModelConfig model_config_;
    uint64_t data_offset_;
    std::vector<TensorPackMapper::PackMapping> tensor_mappings_;
    
    ExpectedVoid read_header() {
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
    
    ExpectedVoid read_metadata() {
        for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
            GGUFMetadataKV kv;
            
            auto str_result = read_string(kv.key);
            if (!str_result.has_value()) {
                return str_result;
            }
            
            file_.read(reinterpret_cast<char*>(&kv.value_type), 4);
            kv.value = read_value_by_type(kv.value_type);
            
            metadata_.push_back(kv);
        }
        
        return make_expected_success();
    }
    
    ExpectedVoid read_tensors() {
        uint64_t current_offset = 0;
        
        for (uint64_t i = 0; i < header_.tensor_count; ++i) {
            GGUFTensorInfo tensor;
            
            auto str_result = read_string(tensor.name);
            if (!str_result.has_value()) {
                return str_result;
            }
            
            file_.read(reinterpret_cast<char*>(&tensor.n_dimensions), 4);
            tensor.dimensions.resize(tensor.n_dimensions);
            
            for (uint32_t j = 0; j < tensor.n_dimensions; ++j) {
                file_.read(reinterpret_cast<char*>(&tensor.dimensions[j]), 8);
            }
            
            file_.read(reinterpret_cast<char*>(&tensor.data_type), 4);
            file_.read(reinterpret_cast<char*>(&tensor.offset), 8);
            
            tensors_.push_back(tensor);
            
            // Track maximum offset for data section
            uint64_t tensor_end = tensor.offset + calculate_tensor_size(tensor);
            current_offset = std::max(current_offset, tensor_end);
        }
        
        // Data section starts after aligned tensor info
        data_offset_ = align_offset(current_offset, 4096);
        
        return make_expected_success();
    }
    
    ExpectedVoid read_string(std::string& str) {
        uint64_t len;
        file_.read(reinterpret_cast<char*>(&len), 8);
        
        str.resize(len);
        file_.read(&str[0], static_cast<std::streamsize>(len));
        
        return make_expected_success();
    }
    
    void* read_value_by_type(GGUFValueType type) {
        void* value = nullptr;
        
        switch (type) {
            case GGUFValueType::UINT8: {
                uint8_t* val = new uint8_t;
                file_.read(reinterpret_cast<char*>(val), 1);
                value = val;
                break;
            }
            case GGUFValueType::INT8: {
                int8_t* val = new int8_t;
                file_.read(reinterpret_cast<char*>(val), 1);
                value = val;
                break;
            }
            case GGUFValueType::UINT16: {
                uint16_t* val = new uint16_t;
                file_.read(reinterpret_cast<char*>(val), 2);
                value = val;
                break;
            }
            case GGUFValueType::INT16: {
                int16_t* val = new int16_t;
                file_.read(reinterpret_cast<char*>(val), 2);
                value = val;
                break;
            }
            case GGUFValueType::UINT32: {
                uint32_t* val = new uint32_t;
                file_.read(reinterpret_cast<char*>(val), 4);
                value = val;
                break;
            }
            case GGUFValueType::INT32: {
                int32_t* val = new int32_t;
                file_.read(reinterpret_cast<char*>(val), 4);
                value = val;
                break;
            }
            case GGUFValueType::FLOAT32: {
                float* val = new float;
                file_.read(reinterpret_cast<char*>(val), 4);
                value = val;
                break;
            }
            case GGUFValueType::UINT64: {
                uint64_t* val = new uint64_t;
                file_.read(reinterpret_cast<char*>(val), 8);
                value = val;
                break;
            }
            case GGUFValueType::INT64: {
                int64_t* val = new int64_t;
                file_.read(reinterpret_cast<char*>(val), 8);
                value = val;
                break;
            }
            case GGUFValueType::FLOAT64: {
                double* val = new double;
                file_.read(reinterpret_cast<char*>(val), 8);
                value = val;
                break;
            }
            case GGUFValueType::BOOL: {
                bool* val = new bool;
                file_.read(reinterpret_cast<char*>(val), 1);
                value = val;
                break;
            }
            case GGUFValueType::STRING: {
                std::string* val = new std::string;
                read_string(*val);
                value = val;
                break;
            }
            case GGUFValueType::ARRAY: {
                // Arrays are complex - read element type and count
                GGUFValueType elem_type;
                uint64_t count;
                file_.read(reinterpret_cast<char*>(&elem_type), 4);
                file_.read(reinterpret_cast<char*>(&count), 8);
                
                // For simplicity, store as raw bytes
                // In production, this would be properly typed
                std::vector<uint8_t>* val = new std::vector<uint8_t>;
                val->resize(count * get_type_size(elem_type));
                file_.read(reinterpret_cast<char*>(val->data()), static_cast<std::streamsize>(val->size()));
                value = val;
                break;
            }
            default:
                value = nullptr;
                break;
        }
        
        return value;
    }
    
    void free_metadata_value(GGUFValueType type, void* value) {
        if (!value) return;
        
        switch (type) {
            case GGUFValueType::UINT8:
            case GGUFValueType::INT8:
            case GGUFValueType::UINT16:
            case GGUFValueType::INT16:
            case GGUFValueType::UINT32:
            case GGUFValueType::INT32:
            case GGUFValueType::FLOAT32:
            case GGUFValueType::UINT64:
            case GGUFValueType::INT64:
            case GGUFValueType::FLOAT64:
            case GGUFValueType::BOOL:
            case GGUFValueType::STRING:
                delete static_cast<std::string*>(value);
                break;
            case GGUFValueType::ARRAY:
                delete static_cast<std::vector<uint8_t>*>(value);
                break;
            default:
                delete[] static_cast<char*>(value);
                break;
        }
    }
    
    void parse_model_config() {
        for (const auto& kv : metadata_) {
            if (kv.key == "general.architecture" && kv.value_type == GGUFValueType::STRING) {
                model_config_.model_type = *static_cast<std::string*>(kv.value);
            } else if (kv.key == "llama.context_length" || kv.key == "llama.contextLength") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.max_position_embeddings = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.embedding_length" || kv.key == "llama.embeddingLength") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.hidden_size = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.feed_forward_length" || kv.key == "llama.feedForwardLength") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.intermediate_size = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.attention.head_count" || kv.key == "llama.attention.headCount") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.num_attention_heads = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.attention.head_count_kv" || kv.key == "llama.attention.headCountKV") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.num_key_value_heads = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.block_count" || kv.key == "llama.blockCount") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.num_layers = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.vocab_size" || kv.key == "llama.vocabSize") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.vocab_size = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.rope.dimension_count" || kv.key == "llama.rope.dimensionCount") {
                if (kv.value_type == GGUFValueType::UINT32) {
                    model_config_.head_dim = *static_cast<uint32_t*>(kv.value);
                }
            } else if (kv.key == "llama.rope.freq_base" || kv.key == "llama.rope.freqBase") {
                if (kv.value_type == GGUFValueType::FLOAT32) {
                    model_config_.rope_theta = *static_cast<float*>(kv.value);
                }
            } else if (kv.key == "llama.attention.layer_norm_rms_epsilon" || kv.key == "llama.attention.layerNormEpsilon") {
                if (kv.value_type == GGUFValueType::FLOAT32) {
                    model_config_.rms_epsilon = *static_cast<float*>(kv.value);
                }
            }
        }
        
        // Calculate head_dim if not provided
        if (model_config_.head_dim == 0 && model_config_.num_attention_heads > 0) {
            model_config_.head_dim = model_config_.hidden_size / model_config_.num_attention_heads;
        }
    }
    
    const GGUFTensorInfo* find_tensor(const std::string& name) const {
        for (const auto& tensor : tensors_) {
            if (tensor.name == name) {
                return &tensor;
            }
        }
        return nullptr;
    }
    
    static uint64_t align_offset(uint64_t offset, uint64_t alignment) {
        return (offset + alignment - 1) / alignment * alignment;
    }
    
    // FP16 to FP32 conversion
    static float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        
        if (exp == 0) {
            if (mant == 0) {
                return sign ? -0.0f : 0.0f;
            }
            // Subnormal
            float val = mant * std::pow(2.0f, -24.0f);
            return sign ? -val : val;
        } else if (exp == 31) {
            if (mant == 0) {
                return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            }
            return std::numeric_limits<float>::quiet_NaN();
        }
        
        // Normal
        float val = std::pow(2.0f, static_cast<float>(exp - 15)) * (1.0f + mant / 1024.0f);
        return sign ? -val : val;
    }
    
    // BF16 to FP32 conversion
    static float bf16_to_fp32(uint16_t b) {
        uint32_t val = static_cast<uint32_t>(b) << 16;
        float result;
        std::memcpy(&result, &val, sizeof(float));
        return result;
    }
    
    // Q8_0 decompression
    std::vector<float> decompress_q8_0(const GGUFTensorInfo& tensor) {
        const uint64_t block_size = 32;
        uint64_t element_count = 1;
        for (uint64_t dim : tensor.dimensions) {
            element_count *= dim;
        }
        
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);
        
        for (uint64_t block = 0; block < num_blocks; ++block) {
            uint64_t block_offset = tensor.offset + block * (sizeof(float) + block_size);
            
            float scale;
            if (!stream_reader_->read_chunk(block_offset, &scale, sizeof(float))) {
                break;
            }
            
            std::vector<int8_t> quantized(block_size);
            if (!stream_reader_->read_chunk(block_offset + sizeof(float), quantized.data(), block_size)) {
                break;
            }
            
            for (uint32_t i = 0; i < block_size && result.size() < element_count; ++i) {
                result.push_back(scale * static_cast<float>(quantized[i]));
            }
        }
        
        return result;
    }
    
    // Q4_0 decompression
    std::vector<float> decompress_q4_0(const GGUFTensorInfo& tensor) {
        const uint64_t block_size = 32;
        uint64_t element_count = 1;
        for (uint64_t dim : tensor.dimensions) {
            element_count *= dim;
        }
        
        uint64_t num_blocks = (element_count + block_size - 1) / block_size;
        std::vector<float> result;
        result.reserve(element_count);
        
        for (uint64_t block = 0; block < num_blocks; ++block) {
            uint64_t block_offset = tensor.offset + block * (sizeof(float) + block_size / 2);
            
            float scale;
            if (!stream_reader_->read_chunk(block_offset, &scale, sizeof(float))) {
                break;
            }
            
            std::vector<uint8_t> quantized(block_size / 2);
            if (!stream_reader_->read_chunk(block_offset + sizeof(float), quantized.data(), block_size / 2)) {
                break;
            }
            
            for (uint32_t i = 0; i < block_size / 2 && result.size() < element_count; ++i) {
                uint8_t packed = quantized[i];
                int8_t q0 = (packed >> 4) & 0x0F;
                int8_t q1 = packed & 0x0F;
                
                // Convert 4-bit to 8-bit range
                result.push_back(scale * static_cast<float>(q0 - 8));
                if (result.size() < element_count) {
                    result.push_back(scale * static_cast<float>(q1 - 8));
                }
            }
        }
        
        return result;
    }
    
    std::vector<float> convert_to_fp32(const std::vector<uint8_t>& data, GGUFValueType type, uint64_t count) {
        std::vector<float> result;
        result.reserve(count);
        
        switch (type) {
            case GGUFValueType::INT8: {
                const int8_t* ptr = reinterpret_cast<const int8_t*>(data.data());
                for (uint64_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            case GGUFValueType::INT16: {
                const int16_t* ptr = reinterpret_cast<const int16_t*>(data.data());
                for (uint64_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            case GGUFValueType::INT32: {
                const int32_t* ptr = reinterpret_cast<const int32_t*>(data.data());
                for (uint64_t i = 0; i < count; ++i) {
                    result.push_back(static_cast<float>(ptr[i]));
                }
                break;
            }
            default:
                // Default: interpret as float
                const float* ptr = reinterpret_cast<const float*>(data.data());
                for (uint64_t i = 0; i < std::min(count, data.size() / sizeof(float)); ++i) {
                    result.push_back(ptr[i]);
                }
                break;
        }
        
        return result;
    }
};

// Public API implementation using PIMPL pattern
GGUFLoader::GGUFLoader(const Path& file_path) 
    : file_path_(file_path), pimpl_(std::make_unique<GGUFLoaderImpl>(file_path)) {}

GGUFLoader::~GGUFLoader() = default;

ExpectedVoid GGUFLoader::load() {
    return pimpl_->load();
}

ExpectedVoid GGUFLoader::close() {
    return pimpl_->close();
}

std::vector<PackMetadata> GGUFLoader::generate_packs(const ModelConfig& config) {
    (void)config; // Config already parsed during load
    return pimpl_->generate_packs();
}

uint64_t GGUFLoader::tensor_count() const {
    return pimpl_->tensors().size();
}

uint64_t GGUFLoader::metadata_count() const {
    return pimpl_->metadata().size();
}

const GGUFTensorInfo* GGUFLoader::get_tensor(const std::string& name) const {
    for (const auto& tensor : pimpl_->tensors()) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

Expected<std::vector<float>> GGUFLoader::read_tensor_data(const GGUFTensorInfo& tensor, bool fp16_to_fp32) {
    (void)tensor; // We look up by name internally
    (void)fp16_to_fp32;
    // For streaming API, use load_tensor_data by name
    return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_FEATURE_NOT_PRESENT));
}

uint64_t GGUFLoader::align_offset(uint64_t offset, uint64_t alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) / alignment * alignment;
}

ModelConfig GGUFLoader::get_model_config() const {
    return pimpl_->model_config();
}

} // namespace vk_symbiote
