#pragma once

#include "VulkanSymbioteEngine.h"

namespace vk_symbiote {

class Tokenizer {
public:
    static std::unique_ptr<Tokenizer> from_gguf(const std::filesystem::path& gguf_path);
    
    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;
    uint32_t vocab_size() const;
    
private:
    std::vector<std::string> vocab_;
    std::vector<uint32_t> token_to_id_;
    std::unordered_map<std::string, uint32_t> token_to_id_map_;
    std::vector<std::string> id_to_token_;
    
    // Helper methods
    static std::unique_ptr<Tokenizer> create_simple_tokenizer();
};

} // namespace vk_symbiote