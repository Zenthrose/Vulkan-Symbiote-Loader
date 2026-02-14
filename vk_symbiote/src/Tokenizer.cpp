#include "Tokenizer.h"
#include "Common.h"
#include "GGUFLoader.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

namespace vk_symbiote {

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const std::filesystem::path& gguf_path) {
    auto tokenizer = std::make_unique<Tokenizer>();
    
    // Try to load vocabulary from GGUF file
    std::ifstream file(gguf_path, std::ios::binary);
    if (!file.is_open()) {
        // Fallback to simple tokenizer if GGUF can't be opened
        return create_simple_tokenizer();
    }
    
    // Read GGUF header to find vocabulary
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "GGUF") {
        file.close();
        return create_simple_tokenizer();
    }
    
    uint32_t version;
    uint64_t tensor_count, metadata_count;
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&tensor_count), 8);
    file.read(reinterpret_cast<char*>(&metadata_count), 8);
    
    // Parse metadata to find vocabulary
    bool found_vocab = false;
    std::unordered_map<std::string, uint32_t> vocab_map;
    std::vector<std::string> vocab_list;
    
    for (uint64_t i = 0; i < metadata_count; ++i) {
        uint64_t key_len, value_type;
        file.read(reinterpret_cast<char*>(&key_len), 8);
        file.read(reinterpret_cast<char*>(&value_type), 4);
        
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);
        
        // Look for vocabulary tokens
        if (key.find("tokenizer.ggml") != std::string::npos) {
            found_vocab = true;
            
            if (value_type == 8) { // STRING type
                uint64_t value_len;
                file.read(reinterpret_cast<char*>(&value_len), 8);
                
                std::string value(value_len, '\0');
                file.read(&value[0], value_len);
                
                // Parse vocabulary entry
                if (key.find("tokens") != std::string::npos) {
                    // Extract token from format like "tokenizer.ggml.tokens.{id}"
                    size_t last_dot = key.find_last_of('.');
                    if (last_dot != std::string::npos) {
                        std::string id_str = key.substr(last_dot + 1);
                        try {
                            uint32_t token_id = std::stoul(id_str);
                            if (token_id < 100000) { // Reasonable vocab size limit
                                vocab_map[value] = token_id;
                                vocab_list.resize(token_id + 1);
                                vocab_list[token_id] = value;
                            }
                        } catch (...) {
                            // Skip invalid token IDs
                        }
                    }
                }
            } else {
                // Skip non-string values
                uint64_t value_len;
                file.read(reinterpret_cast<char*>(&value_len), 8);
                file.seekg(file.tellg() + static_cast<std::streamoff>(value_len));
            }
        } else {
            // Skip unknown metadata entries
            uint64_t value_len;
            file.read(reinterpret_cast<char*>(&value_len), 8);
            file.seekg(file.tellg() + static_cast<std::streamoff>(value_len));
        }
    }
    
    file.close();
    
    if (found_vocab && !vocab_map.empty()) {
        // Build tokenizer from loaded vocabulary
        size_t vocab_size = vocab_list.size();
        tokenizer->vocab_ = vocab_list;
        tokenizer->token_to_id_map_ = vocab_map;
        
        // Reverse map for decoding
        for (const auto& [token, id] : vocab_map) {
            if (id < tokenizer->id_to_token_.size()) {
                tokenizer->id_to_token_[id] = token;
            }
        }
        
        return tokenizer;
    } else {
        // Fallback to simple tokenizer
        return create_simple_tokenizer();
    }
}

std::unique_ptr<Tokenizer> Tokenizer::create_simple_tokenizer() {
    auto tokenizer = std::make_unique<Tokenizer>();
    
    // Create a basic byte-level tokenizer as fallback
    tokenizer->vocab_.resize(256);
    // token_to_id_ is not used in simple tokenizer, only token_to_id_map_
    tokenizer->id_to_token_.resize(256);
    
    for (uint32_t i = 0; i < 256; ++i) {
        std::string token(1, static_cast<char>(i));
        tokenizer->vocab_[i] = token;
        tokenizer->token_to_id_map_[token] = i;
        tokenizer->id_to_token_[i] = token;
    }
    
    return tokenizer;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> tokens;
    
    if (!token_to_id_map_.empty()) {
        // Use vocabulary-based encoding
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            auto it = token_to_id_map_.find(word);
            if (it != token_to_id_map_.end()) {
                tokens.push_back(it->second);
            } else {
                // Try character-level encoding for unknown words
                for (char c : word) {
                    std::string char_str(1, c);
                    auto char_it = token_to_id_map_.find(char_str);
                    if (char_it != token_to_id_map_.end()) {
                        tokens.push_back(char_it->second);
                    } else {
                        // Add as unknown token (hash-based fallback)
                        uint32_t hash = static_cast<uint32_t>(c) * 31;
                        tokens.push_back((hash % 31000) + 1000);
                    }
                }
            }
        }
    } else {
        // Fallback to simple hash-based encoding
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            uint32_t hash = 0;
            for (char c : word) {
                hash = hash * 31 + static_cast<unsigned char>(c);
            }
            uint32_t token_id = (hash % 31000) + 1000;
            tokens.push_back(token_id);
        }
    }
    
    if (tokens.empty()) {
        tokens.push_back(0);
    }
    
    return tokens;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t token_id = tokens[i] % vocab_.size();
        
        if (token_id < vocab_.size()) {
            result += vocab_[token_id];
        } else {
            result += "<unk>";
        }
        
        if (i < tokens.size() - 1) {
            result += " ";
        }
    }
    
    return result;
}

uint32_t Tokenizer::vocab_size() const {
    return static_cast<uint32_t>(vocab_.size());
}

} // namespace vk_symbiote
