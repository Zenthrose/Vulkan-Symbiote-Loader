#include "VitalityOracle.h"
#include "ConfigManager.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstring>
#include <numeric>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <queue>
#include <random>
#include <memory>
#include <chrono>
#include <deque>
#include <array>
#include <functional>

namespace vk_symbiote {

// ============================================================================
// TOML Parser/Writer - Production-Ready Implementation
// ============================================================================
class TOMLParser {
public:
    struct TOMLValue {
        std::variant<std::string, int64_t, double, bool, std::vector<TOMLValue>, std::unordered_map<std::string, TOMLValue>> value;
        
        bool is_string() const { return std::holds_alternative<std::string>(value); }
        bool is_int() const { return std::holds_alternative<int64_t>(value); }
        bool is_float() const { return std::holds_alternative<double>(value); }
        bool is_bool() const { return std::holds_alternative<bool>(value); }
        bool is_array() const { return std::holds_alternative<std::vector<TOMLValue>>(value); }
        bool is_table() const { return std::holds_alternative<std::unordered_map<std::string, TOMLValue>>(value); }
        
        std::string& as_string() { return std::get<std::string>(value); }
        int64_t& as_int() { return std::get<int64_t>(value); }
        double& as_float() { return std::get<double>(value); }
        bool& as_bool() { return std::get<bool>(value); }
        std::vector<TOMLValue>& as_array() { return std::get<std::vector<TOMLValue>>(value); }
        std::unordered_map<std::string, TOMLValue>& as_table() { return std::get<std::unordered_map<std::string, TOMLValue>>(value); }
        
        const std::string& as_string() const { return std::get<std::string>(value); }
        int64_t as_int() const { return std::get<int64_t>(value); }
        double as_float() const { return std::get<double>(value); }
        bool as_bool() const { return std::get<bool>(value); }
        const std::vector<TOMLValue>& as_array() const { return std::get<std::vector<TOMLValue>>(value); }
        const std::unordered_map<std::string, TOMLValue>& as_table() const { return std::get<std::unordered_map<std::string, TOMLValue>>(value); }
    };
    
    using TOMLTable = std::unordered_map<std::string, TOMLValue>;
    using TOMLDocument = std::unordered_map<std::string, TOMLTable>;
    
    static TOMLDocument parse(const std::string& filename) {
        TOMLDocument doc;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[TOMLParser] Failed to open file: " << filename << std::endl;
            return doc;
        }
        
        std::string line;
        std::string current_section;
        int line_num = 0;
        bool in_multi_line_string = false;
        std::string multi_line_buffer;
        
        while (std::getline(file, line)) {
            ++line_num;
            
            // Handle multi-line strings
            if (in_multi_line_string) {
                multi_line_buffer += "\n" + line;
                if (line.find("\"\"\"") != std::string::npos) {
                    in_multi_line_string = false;
                    // Process complete multi-line string
                    if (!current_section.empty()) {
                        size_t eq_pos = multi_line_buffer.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = trim(multi_line_buffer.substr(0, eq_pos));
                            std::string value = extract_multi_line_string(multi_line_buffer.substr(eq_pos + 1));
                            doc[current_section][key] = TOMLValue{value};
                        }
                    }
                    multi_line_buffer.clear();
                }
                continue;
            }
            
            // Trim whitespace
            line = trim(line);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            
            // Parse section header [section] or [[section]] for arrays
            if (line[0] == '[') {
                if (line.size() >= 2 && line[1] == '[') {
                    // Array of tables - simplified handling
                    current_section = extract_section_name(line.substr(2, line.length() - 4));
                } else {
                    current_section = extract_section_name(line.substr(1, line.length() - 2));
                }
                if (doc.find(current_section) == doc.end()) {
                    doc[current_section] = TOMLTable();
                }
                continue;
            }
            
            // Check for multi-line string start
            if (line.find("\"\"\"") != std::string::npos) {
                in_multi_line_string = true;
                multi_line_buffer = line;
                continue;
            }
            
            // Parse key-value pair
            size_t eq_pos = find_equals(line);
            if (eq_pos == std::string::npos) continue;
            
            std::string key = trim(line.substr(0, eq_pos));
            std::string value_str = trim(line.substr(eq_pos + 1));
            
            // Handle inline tables
            if (value_str[0] == '{' && value_str.back() == '}') {
                value_str = value_str.substr(1, value_str.length() - 2);
                TOMLValue table_val;
                table_val.value = parse_inline_table(value_str);
                if (!current_section.empty()) {
                    doc[current_section][key] = table_val;
                }
                continue;
            }
            
            if (current_section.empty()) continue;
            
            doc[current_section][key] = parse_value(value_str);
        }
        
        return doc;
    }
    
    static bool write(const std::string& filename, const TOMLDocument& doc) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[TOMLParser] Failed to write file: " << filename << std::endl;
            return false;
        }
        
        file << "# VitalityOracle Model Configuration\n";
        file << "# Auto-generated by Vulkan Symbiote Engine\n";
        file << "# Timestamp: " << get_timestamp() << "\n\n";
        
        for (const auto& [section, table] : doc) {
            file << "[" << section << "]\n";
            
            // Write non-table values first
            for (const auto& [key, value] : table) {
                if (!value.is_table()) {
                    file << key << " = ";
                    write_value(file, value);
                    file << "\n";
                }
            }
            
            // Write nested tables
            for (const auto& [key, value] : table) {
                if (value.is_table()) {
                    file << "\n[" << section << "." << key << "]\n";
                    write_table(file, value.as_table());
                }
            }
            
            file << "\n";
        }
        
        return file.good();
    }
    
private:
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }
    
    static size_t find_equals(const std::string& str) {
        bool in_string = false;
        for (size_t i = 0; i < str.length(); ++i) {
            if (str[i] == '"' && (i == 0 || str[i-1] != '\\')) {
                in_string = !in_string;
            } else if (str[i] == '=' && !in_string) {
                return i;
            }
        }
        return std::string::npos;
    }
    
    static std::string extract_section_name(const std::string& str) {
        std::string name = trim(str);
        // Remove quotes if present
        if (name.size() >= 2 && name.front() == '"' && name.back() == '"') {
            name = name.substr(1, name.length() - 2);
        }
        return name;
    }
    
    static std::string extract_multi_line_string(const std::string& str) {
        size_t start = str.find("\"\"\"");
        size_t end = str.rfind("\"\"\"");
        if (start != std::string::npos && end != std::string::npos && start != end) {
            return str.substr(start + 3, end - start - 3);
        }
        return str;
    }
    
    static std::unordered_map<std::string, TOMLValue> parse_inline_table(const std::string& str) {
        std::unordered_map<std::string, TOMLValue> result;
        size_t pos = 0;
        
        while (pos < str.length()) {
            size_t comma = find_comma_outside_braces(str, pos);
            std::string pair = trim(str.substr(pos, comma - pos));
            
            size_t eq = pair.find('=');
            if (eq != std::string::npos) {
                std::string key = trim(pair.substr(0, eq));
                std::string value = trim(pair.substr(eq + 1));
                result[key] = parse_value(value);
            }
            
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        
        return result;
    }
    
    static size_t find_comma_outside_braces(const std::string& str, size_t start) {
        int brace_depth = 0;
        bool in_string = false;
        
        for (size_t i = start; i < str.length(); ++i) {
            if (str[i] == '"' && (i == 0 || str[i-1] != '\\')) {
                in_string = !in_string;
            } else if (!in_string) {
                if (str[i] == '{' || str[i] == '[') brace_depth++;
                else if (str[i] == '}' || str[i] == ']') brace_depth--;
                else if (str[i] == ',' && brace_depth == 0) return i;
            }
        }
        return std::string::npos;
    }
    
    static TOMLValue parse_value(const std::string& str) {
        TOMLValue result;
        std::string trimmed = trim(str);
        
        // Try boolean
        if (trimmed == "true") {
            result.value = true;
            return result;
        }
        if (trimmed == "false") {
            result.value = false;
            return result;
        }
        
        // Try array
        if (trimmed[0] == '[' && trimmed.back() == ']') {
            result.value = parse_array(trimmed);
            return result;
        }
        
        // Try string (basic)
        if (trimmed[0] == '"' && trimmed.back() == '"') {
            result.value = parse_string(trimmed);
            return result;
        }
        
        // Try literal string
        if (trimmed[0] == '\'' && trimmed.back() == '\'') {
            result.value = trimmed.substr(1, trimmed.length() - 2);
            return result;
        }
        
        // Try datetime
        if (trimmed.find('T') != std::string::npos || trimmed.find(':') != std::string::npos) {
            // Simplified datetime handling - treat as string
            result.value = trimmed;
            return result;
        }
        
        // Try hex/octal/bin integers
        if (trimmed.size() >= 3 && trimmed[0] == '0') {
            if (trimmed[1] == 'x' || trimmed[1] == 'X') {
                try {
                    result.value = static_cast<int64_t>(std::stoll(trimmed.substr(2), nullptr, 16));
                    return result;
                } catch (...) {}
            } else if (trimmed[1] == 'o' || trimmed[1] == 'O') {
                try {
                    result.value = static_cast<int64_t>(std::stoll(trimmed.substr(2), nullptr, 8));
                    return result;
                } catch (...) {}
            } else if (trimmed[1] == 'b' || trimmed[1] == 'B') {
                try {
                    result.value = static_cast<int64_t>(std::stoll(trimmed.substr(2), nullptr, 2));
                    return result;
                } catch (...) {}
            }
        }
        
        // Try integer
        try {
            size_t pos;
            int64_t int_val = std::stoll(trimmed, &pos);
            if (pos == trimmed.length()) {
                result.value = int_val;
                return result;
            }
        } catch (...) {}
        
        // Try float (including inf and nan)
        if (trimmed == "inf" || trimmed == "+inf") {
            result.value = std::numeric_limits<double>::infinity();
            return result;
        }
        if (trimmed == "-inf") {
            result.value = -std::numeric_limits<double>::infinity();
            return result;
        }
        if (trimmed == "nan" || trimmed == "+nan" || trimmed == "-nan") {
            result.value = std::numeric_limits<double>::quiet_NaN();
            return result;
        }
        
        try {
            size_t pos;
            double float_val = std::stod(trimmed, &pos);
            if (pos == trimmed.length()) {
                result.value = float_val;
                return result;
            }
        } catch (...) {}
        
        // Default to string
        result.value = trimmed;
        return result;
    }
    
    static std::string parse_string(const std::string& str) {
        std::string result;
        result.reserve(str.length());
        
        for (size_t i = 1; i < str.length() - 1; ++i) {
            if (str[i] == '\\' && i + 1 < str.length() - 1) {
                switch (str[i + 1]) {
                    case 'b': result += '\b'; ++i; break;
                    case 't': result += '\t'; ++i; break;
                    case 'n': result += '\n'; ++i; break;
                    case 'f': result += '\f'; ++i; break;
                    case 'r': result += '\r'; ++i; break;
                    case '"': result += '"'; ++i; break;
                    case '\\': result += '\\'; ++i; break;
                    case 'u':
                    case 'U': {
                        // Unicode escape - simplified
                        if (i + 5 < str.length()) {
                            std::string hex = str.substr(i + 2, 4);
                            try {
                                int codepoint = std::stoi(hex, nullptr, 16);
                                result += static_cast<char>(codepoint);
                                i += 5;
                            } catch (...) {
                                result += str[i];
                            }
                        } else {
                            result += str[i];
                        }
                        break;
                    }
                    default: result += str[i]; break;
                }
            } else {
                result += str[i];
            }
        }
        
        return result;
    }
    
    static std::vector<TOMLValue> parse_array(const std::string& str) {
        std::vector<TOMLValue> result;
        
        std::string content = str.substr(1, str.length() - 2);
        size_t pos = 0;
        
        while (pos < content.length()) {
            size_t comma = find_comma_outside_braces(content, pos);
            std::string elem;
            
            if (comma == std::string::npos) {
                elem = trim(content.substr(pos));
                pos = content.length();
            } else {
                elem = trim(content.substr(pos, comma - pos));
                pos = comma + 1;
            }
            
            if (!elem.empty()) {
                result.push_back(parse_value(elem));
            }
        }
        
        return result;
    }
    
    static void write_value(std::ofstream& file, const TOMLValue& value) {
        if (value.is_string()) {
            file << "\"" << escape_string(value.as_string()) << "\"";
        } else if (value.is_int()) {
            file << value.as_int();
        } else if (value.is_float()) {
            double v = value.as_float();
            if (std::isinf(v)) {
                file << (v > 0 ? "inf" : "-inf");
            } else if (std::isnan(v)) {
                file << "nan";
            } else {
                file << std::setprecision(15) << v;
            }
        } else if (value.is_bool()) {
            file << (value.as_bool() ? "true" : "false");
        } else if (value.is_array()) {
            file << "[";
            const auto& arr = value.as_array();
            for (size_t i = 0; i < arr.size(); ++i) {
                write_value(file, arr[i]);
                if (i < arr.size() - 1) file << ", ";
            }
            file << "]";
        } else if (value.is_table()) {
            file << "{";
            const auto& tbl = value.as_table();
            size_t i = 0;
            for (const auto& [k, v] : tbl) {
                file << k << " = ";
                write_value(file, v);
                if (++i < tbl.size()) file << ", ";
            }
            file << "}";
        }
    }
    
    static void write_table(std::ofstream& file, const std::unordered_map<std::string, TOMLValue>& table) {
        for (const auto& [key, value] : table) {
            file << key << " = ";
            write_value(file, value);
            file << "\n";
        }
    }
    
    static std::string escape_string(const std::string& str) {
        std::string result;
        result.reserve(str.length() * 2);
        
        for (char c : str) {
            switch (c) {
                case '\\': result += "\\\\"; break;
                case '"': result += "\\\""; break;
                case '\b': result += "\\b"; break;
                case '\t': result += "\\t"; break;
                case '\n': result += "\\n"; break;
                case '\f': result += "\\f"; break;
                case '\r': result += "\\r"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        result += buf;
                    } else {
                        result += c;
                    }
            }
        }
        
        return result;
    }
    
    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
        return ss.str();
    }
};

// ============================================================================
// Shannon Entropy Calculator for Code Detection
// ============================================================================
class EntropyCalculator {
public:
    // Calculate Shannon entropy of a token sequence
    static float calculate_entropy(const std::vector<uint32_t>& tokens) {
        if (tokens.empty()) return 0.0f;
        
        std::unordered_map<uint32_t, uint32_t> freq;
        for (uint32_t token : tokens) {
            freq[token]++;
        }
        
        float entropy = 0.0f;
        float n = static_cast<float>(tokens.size());
        
        for (const auto& [token, count] : freq) {
            float p = static_cast<float>(count) / n;
            if (p > 0.0f) {
                entropy -= p * std::log2(p);
            }
        }
        
        return entropy;
    }
    
    // Calculate normalized entropy (0.0 to 1.0)
    static float calculate_normalized_entropy(const std::vector<uint32_t>& tokens) {
        if (tokens.empty()) return 0.0f;
        
        float entropy = calculate_entropy(tokens);
        float max_entropy = std::log2(static_cast<float>(tokens.size()));
        
        if (max_entropy <= 0.0f) return 0.0f;
        return std::min(1.0f, entropy / max_entropy);
    }
    
    // Detect if prompt is likely code based on entropy patterns
    static bool is_likely_code(const std::vector<uint32_t>& tokens) {
        if (tokens.size() < 10) return false;
        
        float entropy = calculate_entropy(tokens);
        float normalized = calculate_normalized_entropy(tokens);
        
        // Code typically has moderate entropy (not too random, not too repetitive)
        // Entropy between 2.0 and 5.0 bits per token suggests structured content like code
        // And normalized entropy between 0.3 and 0.8 suggests non-trivial structure
        return (entropy > 2.0f && entropy < 5.0f && normalized > 0.3f && normalized < 0.8f);
    }
    
    // Calculate code-specific scoring bonus
    static float get_code_bonus(const std::vector<uint32_t>& tokens) {
        if (!is_likely_code(tokens)) return 0.0f;
        
        // Calculate additional heuristics
        float complexity = detect_code_complexity(tokens);
        float structure_score = detect_structure_patterns(tokens);
        
        // Boost priority for code-related packs when code is detected
        return 0.15f + complexity * 0.1f + structure_score * 0.05f;
    }
    
    // Detect specific code patterns
    static float detect_code_complexity(const std::vector<uint32_t>& tokens) {
        if (tokens.size() < 20) return 0.0f;
        
        // Look for repetitive patterns that suggest loops/functions
        std::unordered_map<uint32_t, std::vector<uint32_t>> positions;
        for (size_t i = 0; i < tokens.size(); ++i) {
            positions[tokens[i]].push_back(static_cast<uint32_t>(i));
        }
        
        float complexity = 0.0f;
        for (const auto& [token, pos_list] : positions) {
            if (pos_list.size() > 2) {
                // Check for regular spacing (function calls, loops)
                std::vector<float> diffs;
                for (size_t i = 1; i < pos_list.size(); ++i) {
                    diffs.push_back(static_cast<float>(pos_list[i] - pos_list[i-1]));
                }
                
                float avg_diff = std::accumulate(diffs.begin(), diffs.end(), 0.0f) / diffs.size();
                float variance = 0.0f;
                for (float d : diffs) {
                    variance += std::pow(d - avg_diff, 2);
                }
                variance /= diffs.size();
                
                // Low variance in spacing suggests structured code
                if (variance < avg_diff * 0.5f) {
                    complexity += 0.1f;
                }
            }
        }
        
        return std::min(1.0f, complexity);
    }
    
    // Detect structure patterns (indentation-like, brackets, etc.)
    static float detect_structure_patterns(const std::vector<uint32_t>& tokens) {
        if (tokens.size() < 30) return 0.0f;
        
        float score = 0.0f;
        
        // Look for alternating patterns (common in structured code)
        std::unordered_map<uint32_t, uint32_t> pair_counts;
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            uint64_t pair = (static_cast<uint64_t>(tokens[i]) << 32) | tokens[i + 1];
            pair_counts[static_cast<uint32_t>(pair & 0xFFFFFFFF)]++;
        }
        
        // High repetition of specific pairs suggests structure
        uint32_t max_count = 0;
        for (const auto& [pair, count] : pair_counts) {
            max_count = std::max(max_count, count);
        }
        
        float repetition_ratio = static_cast<float>(max_count) / (tokens.size() - 1);
        if (repetition_ratio > 0.05f && repetition_ratio < 0.3f) {
            score += 0.2f;
        }
        
        // Check for hierarchical structure using n-gram analysis
        score += analyze_ngram_structure(tokens);
        
        return std::min(1.0f, score);
    }
    
private:
    static float analyze_ngram_structure(const std::vector<uint32_t>& tokens) {
        if (tokens.size() < 50) return 0.0f;
        
        // Analyze trigram patterns
        std::unordered_map<uint64_t, uint32_t> trigrams;
        for (size_t i = 0; i < tokens.size() - 2; ++i) {
            uint64_t tri = (static_cast<uint64_t>(tokens[i]) << 42) |
                          (static_cast<uint64_t>(tokens[i+1]) << 21) |
                          tokens[i+2];
            trigrams[tri]++;
        }
        
        // Calculate trigram diversity
        float unique_ratio = static_cast<float>(trigrams.size()) / (tokens.size() - 2);
        
        // Code typically has moderate diversity
        if (unique_ratio > 0.3f && unique_ratio < 0.8f) {
            return 0.15f;
        }
        
        return 0.0f;
    }
};

// ============================================================================
// LSTM Cell with Forget/Input/Output Gates and Gradient Tracking
// ============================================================================
struct LSTMCell {
    static constexpr uint32_t INPUT_SIZE = 64;
    static constexpr uint32_t HIDDEN_SIZE = 32;
    static constexpr float WEIGHT_CLIP = 5.0f;  // Prevent exploding gradients
    
    // Gate weights [HIDDEN_SIZE x (INPUT_SIZE + HIDDEN_SIZE)]
    // Order: Forget, Input, Candidate (Cell), Output gates
    std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)> Wf, Wi, Wc, Wo;
    std::array<float, HIDDEN_SIZE> bf, bi, bc, bo;
    
    // Momentum buffers for SGD
    std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)> mWf, mWi, mWc, mWo;
    std::array<float, HIDDEN_SIZE> mbf, mbi, mbc, mbo;
    
    // Velocity buffers for Adam-style optimization (optional)
    std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)> vWf, vWi, vWc, vWo;
    std::array<float, HIDDEN_SIZE> vbf, vbi, vbc, vbo;
    
    // Gradient accumulators
    std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)> gWf, gWi, gWc, gWo;
    std::array<float, HIDDEN_SIZE> gbf, gbi, gbc, gbo;
    
    // Adam optimizer state
    uint32_t timestep = 0;
    
    LSTMCell() : rng_(std::random_device{}()) {
        initialize_weights();
        initialize_momentum();
    }
    
    void initialize_weights() {
        const uint32_t total_input = INPUT_SIZE + HIDDEN_SIZE;
        
        // Xavier/Glorot initialization
        auto init_gate = [&](std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)>& gate, float bias_init = 0.0f) {
            float xavier_scale = std::sqrt(2.0f / (HIDDEN_SIZE + total_input));
            std::normal_distribution<float> dist(0.0f, xavier_scale);
            for (auto& w : gate) w = dist(rng_);
        };
        
        init_gate(Wf, 1.0f);  // Forget gate bias starts at 1.0
        init_gate(Wi);
        init_gate(Wc);
        init_gate(Wo);
        
        // Initialize biases
        bf.fill(1.0f);  // Forget gate: start open
        bi.fill(0.0f);  // Input gate: start closed
        bc.fill(0.0f);  // Candidate: centered
        bo.fill(0.0f);  // Output gate: start closed
    }
    
    void initialize_momentum() {
        mWf.fill(0.0f); mWi.fill(0.0f); mWc.fill(0.0f); mWo.fill(0.0f);
        mbf.fill(0.0f); mbi.fill(0.0f); mbc.fill(0.0f); mbo.fill(0.0f);
        vWf.fill(0.0f); vWi.fill(0.0f); vWc.fill(0.0f); vWo.fill(0.0f);
        vbf.fill(0.0f); vbi.fill(0.0f); vbc.fill(0.0f); vbo.fill(0.0f);
        clear_gradients();
    }
    
    void clear_gradients() {
        gWf.fill(0.0f); gWi.fill(0.0f); gWc.fill(0.0f); gWo.fill(0.0f);
        gbf.fill(0.0f); gbi.fill(0.0f); gbc.fill(0.0f); gbo.fill(0.0f);
    }
    
    // Forward pass with caching for backprop
    struct State {
        std::array<float, HIDDEN_SIZE> h;  // Hidden state
        std::array<float, HIDDEN_SIZE> c;  // Cell state
        
        // Cached intermediate values for backprop
        std::array<float, HIDDEN_SIZE> f_gate, i_gate, c_tilde, o_gate;
        std::array<float, INPUT_SIZE + HIDDEN_SIZE> concat_input;
        std::array<float, HIDDEN_SIZE> tanh_c;  // Cached tanh(c) for output gate
    };
    
    State forward(const std::array<float, INPUT_SIZE>& input, const State& prev_state) {
        State next_state;
        next_state.h.fill(0.0f);
        next_state.c.fill(0.0f);
        
        // Concatenate input and previous hidden state
        std::copy(input.begin(), input.end(), next_state.concat_input.begin());
        std::copy(prev_state.h.begin(), prev_state.h.end(), next_state.concat_input.begin() + INPUT_SIZE);
        
        // Compute all gates in parallel where possible
        compute_forget_gate(next_state);
        compute_input_gate(next_state);
        compute_candidate(next_state);
        compute_cell_state(next_state, prev_state);
        compute_output_gate(next_state);
        compute_hidden_state(next_state);
        
        return next_state;
    }
    
    // Accumulate gradients from backward pass
    void accumulate_gradients(const State& state, const std::array<float, HIDDEN_SIZE>& dh_next,
                              const std::array<float, HIDDEN_SIZE>& dc_next) {
        // Backprop through time - simplified version
        // dh = dh_next + gradients from this step
        std::array<float, HIDDEN_SIZE> dh = dh_next;
        
        // Backprop through output gate
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float do_gate = dh[i] * state.tanh_c[i];
            float do_pre = do_gate * sigmoid_deriv(state.o_gate[i]);
            
            gbo[i] += do_pre;
            for (uint32_t j = 0; j < INPUT_SIZE + HIDDEN_SIZE; ++j) {
                gWo[i * (INPUT_SIZE + HIDDEN_SIZE) + j] += do_pre * state.concat_input[j];
            }
            
            dh[i] = dh[i] * state.o_gate[i] * tanh_deriv(state.tanh_c[i]);
        }
        
        // Add incoming cell gradient
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            dh[i] += dc_next[i];
        }
        
        // Backprop through cell state
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            // Gradient through input gate
            float di_gate = dh[i] * state.c_tilde[i];
            float di_pre = di_gate * sigmoid_deriv(state.i_gate[i]);
            gbi[i] += di_pre;
            
            // Gradient through candidate
            float dc_tilde = dh[i] * state.i_gate[i];
            float dc_pre = dc_tilde * tanh_deriv(state.c_tilde[i]);
            gbc[i] += dc_pre;
            
            for (uint32_t j = 0; j < INPUT_SIZE + HIDDEN_SIZE; ++j) {
                gWi[i * (INPUT_SIZE + HIDDEN_SIZE) + j] += di_pre * state.concat_input[j];
                gWc[i * (INPUT_SIZE + HIDDEN_SIZE) + j] += dc_pre * state.concat_input[j];
            }
        }
    }
    
    // Apply SGD with momentum and optional Adam
    void apply_sgd_momentum(float learning_rate, float momentum, bool use_adam = false,
                            float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f) {
        timestep++;
        
        auto update_weights = [&](std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)>& weights,
                                  std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)>& momentum_buf,
                                  std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)>& velocity_buf,
                                  std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)>& gradients,
                                  bool use_adam) {
            if (use_adam) {
                for (size_t i = 0; i < weights.size(); ++i) {
                    // Adam optimizer
                    momentum_buf[i] = beta1 * momentum_buf[i] + (1.0f - beta1) * gradients[i];
                    velocity_buf[i] = beta2 * velocity_buf[i] + (1.0f - beta2) * gradients[i] * gradients[i];
                    
                    float m_hat = momentum_buf[i] / (1.0f - std::pow(beta1, timestep));
                    float v_hat = velocity_buf[i] / (1.0f - std::pow(beta2, timestep));
                    
                    weights[i] += learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                    weights[i] = std::clamp(weights[i], -WEIGHT_CLIP, WEIGHT_CLIP);
                }
            } else {
                // SGD with momentum
                for (size_t i = 0; i < weights.size(); ++i) {
                    momentum_buf[i] = momentum * momentum_buf[i] + learning_rate * gradients[i];
                    weights[i] += momentum_buf[i];
                    weights[i] = std::clamp(weights[i], -WEIGHT_CLIP, WEIGHT_CLIP);
                }
            }
        };
        
        auto update_biases = [&](std::array<float, HIDDEN_SIZE>& biases,
                                 std::array<float, HIDDEN_SIZE>& momentum_buf,
                                 std::array<float, HIDDEN_SIZE>& velocity_buf,
                                 std::array<float, HIDDEN_SIZE>& gradients,
                                 bool use_adam) {
            if (use_adam) {
                for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
                    momentum_buf[i] = beta1 * momentum_buf[i] + (1.0f - beta1) * gradients[i];
                    velocity_buf[i] = beta2 * velocity_buf[i] + (1.0f - beta2) * gradients[i] * gradients[i];
                    
                    float m_hat = momentum_buf[i] / (1.0f - std::pow(beta1, timestep));
                    float v_hat = velocity_buf[i] / (1.0f - std::pow(beta2, timestep));
                    
                    biases[i] += learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            } else {
                for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
                    momentum_buf[i] = momentum * momentum_buf[i] + learning_rate * gradients[i];
                    biases[i] += momentum_buf[i];
                }
            }
        };
        
        update_weights(Wf, mWf, vWf, gWf, use_adam);
        update_weights(Wi, mWi, vWi, gWi, use_adam);
        update_weights(Wc, mWc, vWc, gWc, use_adam);
        update_weights(Wo, mWo, vWo, gWo, use_adam);
        
        update_biases(bf, mbf, vbf, gbf, use_adam);
        update_biases(bi, mbi, vbi, gbi, use_adam);
        update_biases(bc, mbc, vbc, gbc, use_adam);
        update_biases(bo, mbo, vbo, gbo, use_adam);
        
        clear_gradients();
    }
    
    // Get weight statistics for monitoring
    struct WeightStats {
        float mean, std, min, max;
    };
    
    WeightStats get_weight_stats(const std::array<float, HIDDEN_SIZE * (INPUT_SIZE + HIDDEN_SIZE)>& weights) {
        WeightStats stats;
        float sum = 0.0f, sum_sq = 0.0f;
        stats.min = std::numeric_limits<float>::max();
        stats.max = std::numeric_limits<float>::lowest();
        
        for (float w : weights) {
            sum += w;
            sum_sq += w * w;
            stats.min = std::min(stats.min, w);
            stats.max = std::max(stats.max, w);
        }
        
        stats.mean = sum / weights.size();
        stats.std = std::sqrt(sum_sq / weights.size() - stats.mean * stats.mean);
        
        return stats;
    }
    
private:
    std::mt19937 rng_;
    
    void compute_forget_gate(State& state) {
        const uint32_t total_input = INPUT_SIZE + HIDDEN_SIZE;
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bf[i];
            for (uint32_t j = 0; j < total_input; ++j) {
                sum += Wf[i * total_input + j] * state.concat_input[j];
            }
            state.f_gate[i] = sigmoid(sum);
        }
    }
    
    void compute_input_gate(State& state) {
        const uint32_t total_input = INPUT_SIZE + HIDDEN_SIZE;
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bi[i];
            for (uint32_t j = 0; j < total_input; ++j) {
                sum += Wi[i * total_input + j] * state.concat_input[j];
            }
            state.i_gate[i] = sigmoid(sum);
        }
    }
    
    void compute_candidate(State& state) {
        const uint32_t total_input = INPUT_SIZE + HIDDEN_SIZE;
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bc[i];
            for (uint32_t j = 0; j < total_input; ++j) {
                sum += Wc[i * total_input + j] * state.concat_input[j];
            }
            state.c_tilde[i] = std::tanh(sum);
        }
    }
    
    void compute_cell_state(State& state, const State& prev_state) {
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            state.c[i] = state.f_gate[i] * prev_state.c[i] + state.i_gate[i] * state.c_tilde[i];
        }
    }
    
    void compute_output_gate(State& state) {
        const uint32_t total_input = INPUT_SIZE + HIDDEN_SIZE;
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bo[i];
            for (uint32_t j = 0; j < total_input; ++j) {
                sum += Wo[i * total_input + j] * state.concat_input[j];
            }
            state.o_gate[i] = sigmoid(sum);
        }
    }
    
    void compute_hidden_state(State& state) {
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            state.tanh_c[i] = std::tanh(state.c[i]);
            state.h[i] = state.o_gate[i] * state.tanh_c[i];
        }
    }
    
    static float sigmoid(float x) {
        if (x >= 0) {
            float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        } else {
            float z = std::exp(x);
            return z / (1.0f + z);
        }
    }
    
    static float sigmoid_deriv(float sigmoid_out) {
        return sigmoid_out * (1.0f - sigmoid_out);
    }
    
    static float tanh_deriv(float tanh_out) {
        return 1.0f - tanh_out * tanh_out;
    }
};

// ============================================================================
// Output Layer with Confidence Prediction
// ============================================================================
struct OutputLayer {
    static constexpr uint32_t INPUT_SIZE = LSTMCell::HIDDEN_SIZE;
    static constexpr uint32_t OUTPUT_SIZE = 1;  // Binary: will pack be needed?
    
    std::array<float, OUTPUT_SIZE * INPUT_SIZE> W;
    std::array<float, OUTPUT_SIZE> b;
    
    // Momentum buffers
    std::array<float, OUTPUT_SIZE * INPUT_SIZE> mW;
    std::array<float, OUTPUT_SIZE> mb;
    std::array<float, OUTPUT_SIZE * INPUT_SIZE> vW;
    std::array<float, OUTPUT_SIZE> vb;
    
    uint32_t timestep = 0;
    
    OutputLayer() : rng_(std::random_device{}()) {
        initialize_weights();
    }
    
    void initialize_weights() {
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for (auto& w : W) w = dist(rng_);
        b.fill(0.0f);
        mW.fill(0.0f);
        mb.fill(0.0f);
        vW.fill(0.0f);
        vb.fill(0.0f);
    }
    
    float forward(const std::array<float, INPUT_SIZE>& hidden) {
        float sum = b[0];
        for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
            sum += W[i] * hidden[i];
        }
        return sigmoid(sum);
    }
    
    // Compute gradients for hidden state
    std::array<float, INPUT_SIZE> backward(float prediction, float target,
                                           const std::array<float, INPUT_SIZE>& hidden,
                                           float learning_rate, float momentum) {
        // Binary cross-entropy gradient
        float error = target - prediction;
        float sigmoid_deriv = prediction * (1.0f - prediction);
        float gradient = error * sigmoid_deriv;
        
        // Accumulate gradients for weights
        for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
            mW[i] = momentum * mW[i] + learning_rate * gradient * hidden[i];
            W[i] += mW[i];
        }
        mb[0] = momentum * mb[0] + learning_rate * gradient;
        b[0] += mb[0];
        
        // Return gradient for hidden state
        std::array<float, INPUT_SIZE> dh;
        for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
            dh[i] = gradient * W[i];
        }
        return dh;
    }
    
    float compute_loss(float prediction, float target) {
        // Binary cross-entropy
        float epsilon = 1e-7f;
        prediction = std::clamp(prediction, epsilon, 1.0f - epsilon);
        return -(target * std::log(prediction) + (1.0f - target) * std::log(1.0f - prediction));
    }
    
private:
    std::mt19937 rng_;
    
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-std::clamp(x, -10.0f, 10.0f)));
    }
};

// ============================================================================
// Per-Pack LSTM State with Temporal Tracking
// ============================================================================
struct PackLSTMState {
    LSTMCell::State state;
    uint64_t last_access_time;
    uint64_t access_count;
    std::deque<std::array<float, LSTMCell::INPUT_SIZE>> recent_features;
    static constexpr uint32_t MAX_HISTORY = 10;
    
    // LSTM cell for this pack
    std::unique_ptr<LSTMCell> lstm_cell;
    
    PackLSTMState() : last_access_time(0), access_count(0) {
        state.h.fill(0.0f);
        state.c.fill(0.0f);
        lstm_cell = std::make_unique<LSTMCell>();
    }
    
    void update(const std::array<float, LSTMCell::INPUT_SIZE>& features) {
        // Keep recent feature history
        recent_features.push_back(features);
        if (recent_features.size() > MAX_HISTORY) {
            recent_features.pop_front();
        }
        
        // Update LSTM state
        state = lstm_cell->forward(features, state);
        last_access_time = get_current_time_ns();
        access_count++;
    }
    
    LSTMCell* get_cell() { return lstm_cell.get(); }
    
    float get_temporal_decay(float decay_seconds = 60.0f) const {
        uint64_t age_ns = get_current_time_ns() - last_access_time;
        float age_sec = static_cast<float>(age_ns) / 1e9f;
        return std::exp(-age_sec / decay_seconds);
    }
};

// ============================================================================
// Training Sample Buffer with Priority Replay
// ============================================================================
struct TrainingBuffer {
    struct Sample {
        std::array<float, LSTMCell::INPUT_SIZE> features;
        LSTMCell::State hidden_state;
        float target;
        float predicted;
        float loss;
        uint64_t timestamp;
        uint32_t priority;
    };
    
    std::deque<Sample> samples;
    static constexpr uint32_t MAX_SAMPLES = 2000;
    static constexpr uint32_t MINI_BATCH_SIZE = 32;
    
    // Priority sampling weights
    std::vector<float> priorities;
    
    void add(const std::array<float, LSTMCell::INPUT_SIZE>& features,
             const LSTMCell::State& hidden, float target, float predicted) {
        Sample sample;
        sample.features = features;
        sample.hidden_state = hidden;
        sample.target = target;
        sample.predicted = predicted;
        sample.loss = std::abs(target - predicted);
        sample.timestamp = get_current_time_ns();
        sample.priority = static_cast<uint32_t>(sample.loss * 1000);  // Higher loss = higher priority
        
        samples.push_back(sample);
        if (samples.size() > MAX_SAMPLES) {
            samples.pop_front();
        }
        
        update_priorities();
    }
    
    std::vector<Sample> sample_mini_batch(uint32_t batch_size = MINI_BATCH_SIZE) {
        std::vector<Sample> batch;
        batch.reserve(std::min(batch_size, static_cast<uint32_t>(samples.size())));
        
        if (samples.empty()) return batch;
        
        // Priority-based sampling
        std::vector<uint32_t> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Sort by priority (descending)
        std::sort(indices.begin(), indices.end(), [this](uint32_t a, uint32_t b) {
            return priorities[a] > priorities[b];
        });
        
        // Take top samples
        uint32_t n = std::min(batch_size, static_cast<uint32_t>(indices.size()));
        for (uint32_t i = 0; i < n; ++i) {
            batch.push_back(samples[indices[i]]);
        }
        
        return batch;
    }
    
    void clear() {
        samples.clear();
        priorities.clear();
    }
    
    float get_average_loss() const {
        if (samples.empty()) return 0.0f;
        float total = 0.0f;
        for (const auto& s : samples) total += s.loss;
        return total / samples.size();
    }
    
private:
    void update_priorities() {
        priorities.resize(samples.size());
        size_t i = 0;
        for (const auto& s : samples) {
            priorities[i++] = s.priority + 1.0f;  // Add 1 to ensure non-zero
        }
    }
};

// ============================================================================
// Full VitalityOracle Implementation
// ============================================================================
class VitalityOracleImpl {
public:
    explicit VitalityOracleImpl(uint32_t max_packs_in_memory)
        : max_packs_in_memory_(max_packs_in_memory),
          rng_(std::random_device{}()),
          base_learning_rate_(0.001f),
          momentum_(0.9f),
          sgd_enabled_(true),
          use_adam_(false),
          training_steps_(0),
          hit_rate_(0.0f),
          total_predictions_(0),
          correct_predictions_(0) {
        
        // Load configuration
        auto& cfg = ConfigManager::instance();
        base_learning_rate_ = cfg.get_float("vitality_oracle", "learning_rate", 0.001f);
        momentum_ = cfg.get_float("vitality_oracle", "momentum", 0.9f);
        sgd_enabled_ = cfg.get_bool("vitality_oracle", "sgd_enabled", true);
        use_adam_ = cfg.get_bool("vitality_oracle", "use_adam", false);
        
        // Load model if exists
        std::string model_path = cfg.get_string("vitality_oracle", "model_path", "vitality_oracle_model.toml");
        if (std::filesystem::exists(model_path)) {
            load_model(model_path);
        }
    }
    
    VitalityScore score_pack(const PackMetadata& pack, const std::vector<uint32_t>& tokens,
                            uint32_t current_layer, const HardwareTelemetry& telemetry) {
        VitalityScore score;
        
        // Extract features from current context
        auto features = extract_features(pack, tokens, current_layer, telemetry);
        
        // Get or create LSTM state for this pack
        auto& pack_state = get_or_create_pack_state(pack.pack_id);
        
        // Update LSTM state with current features
        pack_state.update(features);
        
        // Compute component scores
        score.relevance = compute_relevance(pack, tokens, current_layer);
        score.hardware_score = compute_hardware_score(pack, telemetry);
        score.temporal_score = compute_temporal_score(pack.pack_id);
        score.priority_bonus = compute_priority_bonus(pack);
        
        // Entropy-based code detection
        float code_bonus = EntropyCalculator::get_code_bonus(tokens);
        float code_complexity = EntropyCalculator::detect_code_complexity(tokens);
        
        if (code_bonus > 0.0f) {
            // Boost priority for attention-related packs during code generation
            if (pack.type == PackType::ATTENTION_Q || pack.type == PackType::ATTENTION_K ||
                pack.type == PackType::ATTENTION_V) {
                score.priority_bonus += code_bonus + code_complexity * 0.1f;
            }
        }
        
        // LSTM-based confidence prediction
        score.confidence = output_layer_.forward(pack_state.state.h);
        score.confidence = std::clamp(score.confidence, 0.0f, 1.0f);
        
        // Store for training
        last_features_[pack.pack_id] = features;
        last_hidden_[pack.pack_id] = pack_state.state;
        last_prediction_[pack.pack_id] = score.confidence;
        
        return score;
    }
    
    std::vector<uint64_t> predict_next_packs(const std::vector<uint32_t>& tokens,
                                             uint32_t current_layer, uint32_t lookahead, uint32_t count) {
        std::vector<std::pair<uint64_t, float>> scored_packs;
        
        // Generate candidates for next layers
        for (uint32_t offset = 0; offset < lookahead; ++offset) {
            uint32_t layer = current_layer + offset;
            if (layer >= 256) break;  // Sanity limit
            
            float layer_decay = 1.0f - static_cast<float>(offset) / lookahead * 0.5f;
            
            // Score all packs
            for (const auto& [pack_id, pack_state] : pack_states_) {
                // Forward prediction using LSTM
                float predicted_score = output_layer_.forward(pack_state.state.h);
                predicted_score *= layer_decay;
                
                // Add temporal decay based on last access
                float age_score = pack_state.get_temporal_decay();
                predicted_score *= age_score;
                
                // Add exploration bonus for rarely accessed packs
                float exploration_bonus = 0.05f / (1.0f + static_cast<float>(pack_state.access_count) / 100.0f);
                predicted_score += exploration_bonus;
                
                scored_packs.push_back({pack_id, predicted_score});
            }
        }
        
        // Sort by score descending
        std::sort(scored_packs.begin(), scored_packs.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top predictions
        std::vector<uint64_t> predictions;
        predictions.reserve(std::min(count, static_cast<uint32_t>(scored_packs.size())));
        
        for (uint32_t i = 0; i < count && i < scored_packs.size(); ++i) {
            predictions.push_back(scored_packs[i].first);
        }
        
        return predictions;
    }
    
    void record_access(uint64_t pack_id, bool was_used, float predicted_score) {
        AccessRecord record;
        record.pack_id = pack_id;
        record.timestamp = get_current_time_ns();
        record.was_used = was_used;
        record.predicted_score = predicted_score;
        
        recent_accesses_.push_back(record);
        if (recent_accesses_.size() > max_access_history_) {
            recent_accesses_.pop_front();
        }
        
        // Update statistics
        total_predictions_++;
        if (was_used) {
            correct_predictions_++;
        }
        hit_rate_ = static_cast<float>(correct_predictions_) / static_cast<float>(total_predictions_);
        
        // Add to training buffer
        auto it = last_features_.find(pack_id);
        auto hidden_it = last_hidden_.find(pack_id);
        if (it != last_features_.end() && hidden_it != last_hidden_.end()) {
            float target = was_used ? 1.0f : 0.0f;
            training_buffer_.add(it->second, hidden_it->second, target, predicted_score);
        }
    }
    
    void update_model() {
        if (training_buffer_.samples.size() < TrainingBuffer::MINI_BATCH_SIZE) return;
        
        // Adaptive learning rate decay
        float learning_rate = base_learning_rate_ *
            (1.0f / (1.0f + 0.0001f * static_cast<float>(training_steps_)));
        
        // Sample mini-batch with priority
        auto batch = training_buffer_.sample_mini_batch();
        
        float total_loss = 0.0f;
        
        // Train output layer
        for (const auto& sample : batch) {
            float prediction = output_layer_.forward(sample.hidden_state.h);
            float loss = output_layer_.compute_loss(prediction, sample.target);
            total_loss += loss;
            
            // Backprop through output layer
            auto dh = output_layer_.backward(prediction, sample.target,
                                             sample.hidden_state.h, learning_rate, momentum_);
            
            // Accumulate gradients in corresponding LSTM cell
            auto& pack_state = get_or_create_pack_state(sample.timestamp);  // Use timestamp as proxy ID
            if (LSTMCell* cell = pack_state.get_cell()) {
                std::array<float, LSTMCell::HIDDEN_SIZE> dc_next;
                dc_next.fill(0.0f);
                cell->accumulate_gradients(sample.hidden_state, dh, dc_next);
            }
        }
        
        // Apply updates to all LSTM cells
        for (auto& [pack_id, pack_state] : pack_states_) {
            if (LSTMCell* cell = pack_state.get_cell()) {
                cell->apply_sgd_momentum(learning_rate, momentum_, use_adam_);
            }
        }
        
        training_steps_++;
        
        // Log training stats periodically
        if (training_steps_ % 100 == 0) {
            float avg_loss = total_loss / batch.size();
            float avg_buffer_loss = training_buffer_.get_average_loss();
            std::cout << "[VitalityOracle] Training step " << training_steps_
                      << ", avg_loss: " << avg_loss
                      << ", buffer_loss: " << avg_buffer_loss
                      << ", hit_rate: " << hit_rate_
                      << ", lr: " << learning_rate
                      << ", momentum: " << momentum_
                      << ", optimizer: " << (use_adam_ ? "Adam" : "SGD")
                      << std::endl;
        }
        
        // Clear buffer after training
        training_buffer_.clear();
    }
    
    void save_model(const std::string& path) {
        TOMLParser::TOMLDocument doc;
        
        // Header section
        doc["header"]["version"] = TOMLParser::TOMLValue{1};
        doc["header"]["training_steps"] = TOMLParser::TOMLValue{static_cast<int64_t>(training_steps_)};
        doc["header"]["hit_rate"] = TOMLParser::TOMLValue{static_cast<double>(hit_rate_)};
        doc["header"]["total_predictions"] = TOMLParser::TOMLValue{static_cast<int64_t>(total_predictions_)};
        doc["header"]["correct_predictions"] = TOMLParser::TOMLValue{static_cast<int64_t>(correct_predictions_)};
        doc["header"]["learning_rate"] = TOMLParser::TOMLValue{static_cast<double>(base_learning_rate_)};
        doc["header"]["momentum"] = TOMLParser::TOMLValue{static_cast<double>(momentum_)};
        doc["header"]["sgd_enabled"] = TOMLParser::TOMLValue{sgd_enabled_};
        doc["header"]["use_adam"] = TOMLParser::TOMLValue{use_adam_};
        
        // Output layer weights
        TOMLParser::TOMLValue W_array;
        W_array.value = std::vector<TOMLParser::TOMLValue>();
        for (float w : output_layer_.W) {
            W_array.as_array().push_back(TOMLParser::TOMLValue{static_cast<double>(w)});
        }
        doc["output_layer"]["W"] = W_array;
        doc["output_layer"]["b"] = TOMLParser::TOMLValue{static_cast<double>(output_layer_.b[0])};
        
        // Save pack-specific LSTM states
        for (const auto& [pack_id, pack_state] : pack_states_) {
            if (LSTMCell* cell = const_cast<PackLSTMState&>(pack_state).get_cell()) {
                std::string section = "lstm.pack_" + std::to_string(pack_id);
                save_lstm_weights(doc, section, *cell);
                
                // Save state
                TOMLParser::TOMLValue h_array, c_array;
                h_array.value = std::vector<TOMLParser::TOMLValue>();
                c_array.value = std::vector<TOMLParser::TOMLValue>();
                
                for (float v : pack_state.state.h) {
                    h_array.as_array().push_back(TOMLParser::TOMLValue{static_cast<double>(v)});
                }
                for (float v : pack_state.state.c) {
                    c_array.as_array().push_back(TOMLParser::TOMLValue{static_cast<double>(v)});
                }
                
                doc[section]["hidden_state"] = h_array;
                doc[section]["cell_state"] = c_array;
                doc[section]["access_count"] = TOMLParser::TOMLValue{static_cast<int64_t>(pack_state.access_count)};
            }
        }
        
        if (TOMLParser::write(path, doc)) {
            std::cout << "[VitalityOracle] Model saved to TOML: " << path << std::endl;
        } else {
            std::cerr << "[VitalityOracle] Failed to save model to: " << path << std::endl;
        }
    }
    
    void load_model(const std::string& path) {
        TOMLParser::TOMLDocument doc = TOMLParser::parse(path);
        
        if (doc.empty()) {
            std::cerr << "[VitalityOracle] Failed to load model from: " << path << std::endl;
            return;
        }
        
        // Parse header
        if (doc.find("header") != doc.end()) {
            const auto& header = doc["header"];
            
            auto it = header.find("training_steps");
            if (it != header.end() && it->second.is_int()) {
                training_steps_ = static_cast<uint32_t>(it->second.as_int());
            }
            
            it = header.find("hit_rate");
            if (it != header.end() && it->second.is_float()) {
                hit_rate_ = static_cast<float>(it->second.as_float());
            }
            
            it = header.find("total_predictions");
            if (it != header.end() && it->second.is_int()) {
                total_predictions_ = static_cast<uint64_t>(it->second.as_int());
            }
            
            it = header.find("correct_predictions");
            if (it != header.end() && it->second.is_int()) {
                correct_predictions_ = static_cast<uint64_t>(it->second.as_int());
            }
            
            it = header.find("learning_rate");
            if (it != header.end() && it->second.is_float()) {
                base_learning_rate_ = static_cast<float>(it->second.as_float());
            }
            
            it = header.find("momentum");
            if (it != header.end() && it->second.is_float()) {
                momentum_ = static_cast<float>(it->second.as_float());
            }
            
            it = header.find("sgd_enabled");
            if (it != header.end() && it->second.is_bool()) {
                sgd_enabled_ = it->second.as_bool();
            }
            
            it = header.find("use_adam");
            if (it != header.end() && it->second.is_bool()) {
                use_adam_ = it->second.as_bool();
            }
        }
        
        // Parse output layer
        if (doc.find("output_layer") != doc.end()) {
            const auto& output = doc["output_layer"];
            
            auto it = output.find("W");
            if (it != output.end() && it->second.is_array()) {
                const auto& W_array = it->second.as_array();
                for (size_t i = 0; i < W_array.size() && i < output_layer_.W.size(); ++i) {
                    if (W_array[i].is_float()) {
                        output_layer_.W[i] = static_cast<float>(W_array[i].as_float());
                    }
                }
            }
            
            it = output.find("b");
            if (it != output.end() && it->second.is_float()) {
                output_layer_.b[0] = static_cast<float>(it->second.as_float());
            }
        }
        
        // Parse pack-specific LSTM weights and states
        for (const auto& [section, table] : doc) {
            if (section.find("lstm.pack_") == 0) {
                uint64_t pack_id = std::stoull(section.substr(10));
                
                auto& pack_state = get_or_create_pack_state(pack_id);
            if (LSTMCell* cell = const_cast<PackLSTMState&>(pack_state).get_cell()) {
                    load_lstm_weights(table, *cell);
                    
                    // Load state if available
                    auto h_it = table.find("hidden_state");
                    if (h_it != table.end() && h_it->second.is_array()) {
                        const auto& h_array = h_it->second.as_array();
                        for (size_t i = 0; i < h_array.size() && i < LSTMCell::HIDDEN_SIZE; ++i) {
                            if (h_array[i].is_float()) {
                                pack_state.state.h[i] = static_cast<float>(h_array[i].as_float());
                            }
                        }
                    }
                    
                    auto c_it = table.find("cell_state");
                    if (c_it != table.end() && c_it->second.is_array()) {
                        const auto& c_array = c_it->second.as_array();
                        for (size_t i = 0; i < c_array.size() && i < LSTMCell::HIDDEN_SIZE; ++i) {
                            if (c_array[i].is_float()) {
                                pack_state.state.c[i] = static_cast<float>(c_array[i].as_float());
                            }
                        }
                    }
                    
                    auto acc_it = table.find("access_count");
                    if (acc_it != table.end() && acc_it->second.is_int()) {
                        pack_state.access_count = static_cast<uint64_t>(acc_it->second.as_int());
                    }
                }
            }
        }
        
        std::cout << "[VitalityOracle] Model loaded from TOML: " << path
                  << " (training_steps: " << training_steps_ << ")" << std::endl;
    }
    
    // Configuration
    void set_learning_rate(float lr) { base_learning_rate_ = lr; }
    float get_learning_rate() const { return base_learning_rate_; }
    void set_momentum(float m) { momentum_ = m; }
    float get_momentum() const { return momentum_; }
    void enable_sgd(bool enable) { sgd_enabled_ = enable; }
    bool is_sgd_enabled() const { return sgd_enabled_; }
    void enable_adam(bool enable) { use_adam_ = enable; }
    bool is_adam_enabled() const { return use_adam_; }
    
    // Statistics
    float hit_rate() const { return hit_rate_; }
    uint64_t total_predictions() const { return total_predictions_; }
    uint32_t get_training_steps() const { return training_steps_; }
    
    // Get LSTM weight statistics for debugging
    void print_lstm_stats(uint64_t pack_id) {
        auto it = pack_states_.find(pack_id);
        if (it == pack_states_.end()) {
            std::cout << "[VitalityOracle] Pack " << pack_id << " not found" << std::endl;
            return;
        }
        
        if (LSTMCell* cell = it->second.get_cell()) {
            std::cout << "[VitalityOracle] LSTM stats for pack " << pack_id << ":" << std::endl;
            
            auto wf_stats = cell->get_weight_stats(cell->Wf);
            std::cout << "  Forget gate weights: mean=" << wf_stats.mean
                      << " std=" << wf_stats.std
                      << " min=" << wf_stats.min
                      << " max=" << wf_stats.max << std::endl;
        }
    }
    
private:
    uint32_t max_packs_in_memory_;
    std::mt19937 rng_;
    
    std::unordered_map<uint64_t, PackLSTMState> pack_states_;
    OutputLayer output_layer_;
    
    std::deque<AccessRecord> recent_accesses_;
    static constexpr uint32_t max_access_history_ = 10000;
    TrainingBuffer training_buffer_;
    
    std::unordered_map<uint64_t, std::array<float, LSTMCell::INPUT_SIZE>> last_features_;
    std::unordered_map<uint64_t, LSTMCell::State> last_hidden_;
    std::unordered_map<uint64_t, float> last_prediction_;
    
    float base_learning_rate_;
    float momentum_;
    bool sgd_enabled_;
    bool use_adam_;
    uint32_t training_steps_;
    
    float hit_rate_;
    uint64_t total_predictions_;
    uint64_t correct_predictions_;
    
    PackLSTMState& get_or_create_pack_state(uint64_t pack_id) {
        auto it = pack_states_.find(pack_id);
        if (it == pack_states_.end()) {
            it = pack_states_.emplace(pack_id, PackLSTMState()).first;
        }
        return it->second;
    }
    
    std::array<float, LSTMCell::INPUT_SIZE> extract_features(const PackMetadata& pack,
                                                              const std::vector<uint32_t>& tokens,
                                                              uint32_t current_layer,
                                                              const HardwareTelemetry& telemetry) {
        std::array<float, LSTMCell::INPUT_SIZE> features;
        features.fill(0.0f);
        
        // Pack metadata features
        features[0] = static_cast<float>(pack.layer_idx) / 128.0f;
        features[1] = static_cast<float>(pack.head_idx) / 64.0f;
        features[2] = static_cast<float>(static_cast<int>(pack.type)) / 255.0f;
        features[3] = pack.base_priority;
        
        // Context features
        features[4] = static_cast<float>(current_layer) / 128.0f;
        features[5] = static_cast<float>(tokens.size()) / 8192.0f;
        
        // Hardware features
        features[6] = telemetry.gpu_utilization;
        features[7] = telemetry.memory_bandwidth;
        features[8] = telemetry.cache_hit_rate;
        features[9] = telemetry.compute_occupancy;
        features[10] = telemetry.thermal_throttle;
        
        // Temporal features
        auto& pack_state = get_or_create_pack_state(pack.pack_id);
        features[11] = static_cast<float>(pack_state.recent_features.size()) / PackLSTMState::MAX_HISTORY;
        features[12] = static_cast<float>(pack_state.access_count) / 1000.0f;
        
        // Token hash features (locality-sensitive hashing)
        uint64_t token_hash = hash_tokens(tokens);
        for (uint32_t i = 0; i < 8; ++i) {
            features[13 + i] = static_cast<float>((token_hash >> (i * 8)) & 0xFF) / 255.0f;
        }
        
        // Computed scores as features
        features[21] = compute_relevance(pack, tokens, current_layer);
        features[22] = compute_hardware_score(pack, telemetry);
        features[23] = compute_temporal_score(pack.pack_id);
        
        // Time-based cyclical features
        uint64_t time_ms = get_current_time_ms();
        float time_sec = static_cast<float>(time_ms % 1000000) / 1000.0f;
        features[24] = std::sin(2.0f * 3.14159f * time_sec / 60.0f);
        features[25] = std::cos(2.0f * 3.14159f * time_sec / 60.0f);
        
        // Pack size features
        features[26] = static_cast<float>(pack.compressed_size) / (1024.0f * 1024.0f * 1024.0f);
        features[27] = static_cast<float>(pack.decompressed_size) / (1024.0f * 1024.0f * 1024.0f);
        
        // Entropy features
        float entropy = EntropyCalculator::calculate_entropy(tokens);
        float normalized_entropy = EntropyCalculator::calculate_normalized_entropy(tokens);
        features[28] = entropy / 10.0f;
        features[29] = normalized_entropy;
        features[30] = EntropyCalculator::is_likely_code(tokens) ? 1.0f : 0.0f;
        features[31] = EntropyCalculator::detect_code_complexity(tokens);
        
        // Remaining features with small noise for regularization
        std::uniform_real_distribution<float> noise_dist(-0.01f, 0.01f);
        for (uint32_t i = 32; i < LSTMCell::INPUT_SIZE; ++i) {
            features[i] = noise_dist(rng_);
        }
        
        return features;
    }
    
    uint64_t hash_tokens(const std::vector<uint32_t>& tokens) const {
        uint64_t hash = 1469598103934665603ULL;
        for (uint32_t token : tokens) {
            hash ^= token;
            hash *= 1099511628211ULL;
        }
        return hash;
    }
    
    float compute_relevance(const PackMetadata& pack, const std::vector<uint32_t>& tokens, uint32_t current_layer) {
        float layer_dist = std::abs(static_cast<float>(pack.layer_idx) - static_cast<float>(current_layer));
        float layer_score = std::exp(-layer_dist / 5.0f);
        
        float token_score = 0.5f;
        if (!tokens.empty()) {
            token_score = 0.5f + 0.3f * std::sin(static_cast<float>(tokens.back()) / 1000.0f);
        }
        
        float type_weight = 0.7f;
        switch (pack.type) {
            case PackType::ATTENTION_Q:
            case PackType::ATTENTION_K:
            case PackType::ATTENTION_V:
                type_weight = 1.0f;
                break;
            case PackType::ATTENTION_O:
                type_weight = 0.9f;
                break;
            case PackType::NORM_GAMMA:
            case PackType::NORM_BETA:
                type_weight = 0.85f;
                break;
            default:
                break;
        }
        
        return layer_score * token_score * type_weight;
    }
    
    float compute_hardware_score(const PackMetadata& pack, const HardwareTelemetry& telemetry) {
        (void)pack;
        
        float score = 1.0f;
        
        if (telemetry.gpu_utilization > 0.9f) {
            score *= 0.6f;
        } else if (telemetry.gpu_utilization < 0.3f) {
            score *= 1.2f;
        }
        
        if (telemetry.memory_bandwidth > 0.85f) {
            score *= 0.7f;
        }
        
        score *= (0.5f + 0.5f * telemetry.cache_hit_rate);
        score *= telemetry.thermal_throttle;
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    float compute_temporal_score(uint64_t pack_id) {
        auto it = pack_states_.find(pack_id);
        if (it == pack_states_.end()) {
            return 0.5f;
        }
        
        return it->second.get_temporal_decay();
    }
    
    float compute_priority_bonus(const PackMetadata& pack) {
        return pack.base_priority * 0.7f +
               (1.0f - static_cast<float>(pack.layer_idx) / 128.0f) * 0.3f;
    }
    
    void save_lstm_weights(TOMLParser::TOMLDocument& doc, const std::string& section, const LSTMCell& cell) {
        auto save_matrix = [&](const std::string& name, const auto& mat) {
            TOMLParser::TOMLValue arr;
            arr.value = std::vector<TOMLParser::TOMLValue>();
            for (float v : mat) {
                arr.as_array().push_back(TOMLParser::TOMLValue{static_cast<double>(v)});
            }
            doc[section][name] = arr;
        };
        
        save_matrix("Wf", cell.Wf);
        save_matrix("Wi", cell.Wi);
        save_matrix("Wc", cell.Wc);
        save_matrix("Wo", cell.Wo);
        save_matrix("bf", cell.bf);
        save_matrix("bi", cell.bi);
        save_matrix("bc", cell.bc);
        save_matrix("bo", cell.bo);
    }
    
    void load_lstm_weights(const TOMLParser::TOMLTable& table, LSTMCell& cell) {
        auto load_matrix = [&](const std::string& name, auto& mat) {
            auto it = table.find(name);
            if (it != table.end() && it->second.is_array()) {
                const auto& arr = it->second.as_array();
                for (size_t i = 0; i < arr.size() && i < mat.size(); ++i) {
                    if (arr[i].is_float()) {
                        mat[i] = static_cast<float>(arr[i].as_float());
                    }
                }
            }
        };
        
        load_matrix("Wf", cell.Wf);
        load_matrix("Wi", cell.Wi);
        load_matrix("Wc", cell.Wc);
        load_matrix("Wo", cell.Wo);
        load_matrix("bf", cell.bf);
        load_matrix("bi", cell.bi);
        load_matrix("bc", cell.bc);
        load_matrix("bo", cell.bo);
    }
};

// ============================================================================
// Public API Implementation
// ============================================================================
VitalityOracle::VitalityOracle(uint32_t max_packs_in_memory)
    : pimpl_(std::make_unique<VitalityOracleImpl>(max_packs_in_memory)) {
}

VitalityOracle::~VitalityOracle() = default;

VitalityScore VitalityOracle::score_pack(const PackMetadata& pack, const std::vector<uint32_t>& tokens,
                                        uint32_t current_layer, const HardwareTelemetry& telemetry) {
    return pimpl_->score_pack(pack, tokens, current_layer, telemetry);
}

std::vector<uint64_t> VitalityOracle::predict_next_packs(const std::vector<uint32_t>& tokens,
                                                         uint32_t current_layer, uint32_t lookahead, uint32_t count) {
    return pimpl_->predict_next_packs(tokens, current_layer, lookahead, count);
}

void VitalityOracle::update_telemetry(const HardwareTelemetry& telemetry) {
    (void)telemetry;
}

void VitalityOracle::record_access(uint64_t pack_id, bool was_used, float predicted_score) {
    pimpl_->record_access(pack_id, was_used, predicted_score);
}

void VitalityOracle::update_model() {
    pimpl_->update_model();
}

float VitalityOracle::hit_rate() const noexcept {
    return pimpl_->hit_rate();
}

uint64_t VitalityOracle::total_predictions() const noexcept {
    return pimpl_->total_predictions();
}

void VitalityOracle::save_model(const std::string& path) {
    pimpl_->save_model(path);
}

void VitalityOracle::load_model(const std::string& path) {
    pimpl_->load_model(path);
}

float VitalityOracle::calculate_token_entropy(const std::vector<uint32_t>& tokens) {
    return EntropyCalculator::calculate_entropy(tokens);
}

bool VitalityOracle::is_code_prompt(const std::vector<uint32_t>& tokens) {
    return EntropyCalculator::is_likely_code(tokens);
}

void VitalityOracle::set_learning_rate(float lr) {
    pimpl_->set_learning_rate(lr);
}

float VitalityOracle::get_learning_rate() const {
    return pimpl_->get_learning_rate();
}

void VitalityOracle::enable_sgd_training(bool enable) {
    pimpl_->enable_sgd(enable);
}

void VitalityOracle::enable_adam_training(bool enable) {
    pimpl_->enable_adam(enable);
}

uint32_t VitalityOracle::get_training_steps() const {
    return pimpl_->get_training_steps();
}

void VitalityOracle::print_lstm_stats(uint64_t pack_id) {
    pimpl_->print_lstm_stats(pack_id);
}

} // namespace vk_symbiote
