#include "ConfigManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <variant>
#include <unordered_map>

namespace vk_symbiote {

std::unique_ptr<ConfigManager> ConfigManager::instance_ = nullptr;
std::mutex ConfigManager::instance_mutex_;

ConfigManager& ConfigManager::instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_unique<ConfigManager>(PrivateTag{});
        instance_->set_defaults();
    }
    return *instance_;
}

// ============================================================================
// Full TOML Parser Implementation
// ============================================================================
class TOMLParser {
public:
    struct TOMLValue {
        std::variant<std::string, int64_t, double, bool, std::vector<TOMLValue>> value;
        
        bool is_string() const { return std::holds_alternative<std::string>(value); }
        bool is_int() const { return std::holds_alternative<int64_t>(value); }
        bool is_float() const { return std::holds_alternative<double>(value); }
        bool is_bool() const { return std::holds_alternative<bool>(value); }
        bool is_array() const { return std::holds_alternative<std::vector<TOMLValue>>(value); }
        
        std::string& as_string() { return std::get<std::string>(value); }
        int64_t& as_int() { return std::get<int64_t>(value); }
        double& as_float() { return std::get<double>(value); }
        bool& as_bool() { return std::get<bool>(value); }
        std::vector<TOMLValue>& as_array() { return std::get<std::vector<TOMLValue>>(value); }
        
        const std::string& as_string() const { return std::get<std::string>(value); }
        int64_t as_int() const { return std::get<int64_t>(value); }
        double as_float() const { return std::get<double>(value); }
        bool as_bool() const { return std::get<bool>(value); }
        const std::vector<TOMLValue>& as_array() const { return std::get<std::vector<TOMLValue>>(value); }
    };
    
    using TOMLTable = std::unordered_map<std::string, TOMLValue>;
    using TOMLDocument = std::unordered_map<std::string, TOMLTable>;
    
    static TOMLDocument parse(const std::string& filename) {
        TOMLDocument doc;
        std::ifstream file(filename);
        if (!file.is_open()) return doc;
        
        std::string line;
        std::string current_section;
        int line_num = 0;
        bool in_multiline_string = false;
        std::string multiline_buffer;
        
        while (std::getline(file, line)) {
            ++line_num;
            
            // Handle multiline strings
            if (in_multiline_string) {
                if (line.find(""""") != std::string::npos) {
                    multiline_buffer += line.substr(0, line.find("""""));
                    in_multiline_string = false;
                    // Process accumulated multiline string
                    if (!current_section.empty()) {
                        // Add to doc[current_section] if we have a key
                    }
                    multiline_buffer.clear();
                } else {
                    multiline_buffer += line + "\n";
                }
                continue;
            }
            
            // Trim whitespace
            line = trim(line);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            
            // Parse section header [section]
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                // Handle nested sections like [section.subsection]
                size_t dot_pos = current_section.find('.');
                if (dot_pos != std::string::npos) {
                    // Convert to flattened section name
                    std::replace(current_section.begin(), current_section.end(), '.', '_');
                }
                if (doc.find(current_section) == doc.end()) {
                    doc[current_section] = TOMLTable();
                }
                continue;
            }
            
            // Parse key-value pair
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;
            
            std::string key = trim(line.substr(0, eq_pos));
            std::string value_str = trim(line.substr(eq_pos + 1));
            
            // Handle multiline string start
            if (value_str.substr(0, 3) == "\"\"\"") {
                in_multiline_string = true;
                multiline_buffer = value_str.substr(3);
                if (multiline_buffer.find(""""")) != std::string::npos) {
                    in_multiline_string = false;
                    value_str = multiline_buffer.substr(0, multiline_buffer.find("""""));
                    multiline_buffer.clear();
                } else {
                    continue;
                }
            }
            
            if (current_section.empty()) continue;
            
            doc[current_section][key] = parse_value(value_str);
        }
        
        return doc;
    }
    
    static bool write(const std::string& filename, const TOMLDocument& doc) {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        
        file << "# Vulkan Symbiote Configuration\n";
        file << "# Auto-generated configuration file\n\n";
        
        for (const auto& [section, table] : doc) {
            file << "[" << section << "]\n";
            
            for (const auto& [key, value] : table) {
                file << key << " = ";
                write_value(file, value);
                file << "\n";
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
        if (!trimmed.empty() && trimmed[0] == '[' && trimmed.back() == ']') {
            result.value = parse_array(trimmed);
            return result;
        }
        
        // Try string (single or double quotes)
        if ((trimmed[0] == '"' && trimmed.back() == '"') ||
            (trimmed[0] == '\'' && trimmed.back() == '\'')) {
            result.value = trimmed.substr(1, trimmed.length() - 2);
            return result;
        }
        
        // Try datetime
        if (trimmed.find('T') != std::string::npos || trimmed.find(':') != std::string::npos) {
            // Treat as string for simplicity
            result.value = trimmed;
            return result;
        }
        
        // Try integer (including hex, octal, binary)
        try {
            size_t pos;
            // Check for hex
            if (trimmed.substr(0, 2) == "0x" || trimmed.substr(0, 2) == "0X") {
                int64_t int_val = std::stoll(trimmed, &pos, 16);
                if (pos == trimmed.length()) {
                    result.value = int_val;
                    return result;
                }
            }
            // Check for binary
            else if (trimmed.substr(0, 2) == "0b" || trimmed.substr(0, 2) == "0B") {
                int64_t int_val = std::stoll(trimmed.substr(2), &pos, 2);
                if (pos == trimmed.length() - 2) {
                    result.value = int_val;
                    return result;
                }
            }
            // Check for octal
            else if (trimmed.length() > 1 && trimmed[0] == '0' && std::isdigit(trimmed[1])) {
                int64_t int_val = std::stoll(trimmed, &pos, 8);
                if (pos == trimmed.length()) {
                    result.value = int_val;
                    return result;
                }
            }
            // Decimal integer
            else {
                int64_t int_val = std::stoll(trimmed, &pos);
                if (pos == trimmed.length()) {
                    result.value = int_val;
                    return result;
                }
            }
        } catch (...) {}
        
        // Try float (including scientific notation, inf, nan)
        try {
            if (trimmed == "inf" || trimmed == "+inf" || trimmed == "-inf" ||
                trimmed == "nan" || trimmed == "+nan" || trimmed == "-nan") {
                result.value = std::stod(trimmed);
                return result;
            }
            
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
    
    static std::vector<TOMLValue> parse_array(const std::string& str) {
        std::vector<TOMLValue> result;
        
        std::string content = str.substr(1, str.length() - 2);
        size_t pos = 0;
        int bracket_depth = 0;
        std::string current_element;
        
        while (pos < content.length()) {
            char c = content[pos];
            
            if (c == '[') {
                bracket_depth++;
                current_element += c;
            } else if (c == ']') {
                bracket_depth--;
                current_element += c;
            } else if (c == ',' && bracket_depth == 0) {
                std::string elem = trim(current_element);
                if (!elem.empty()) {
                    result.push_back(parse_value(elem));
                }
                current_element.clear();
            } else if (c == '"') {
                // Handle quoted strings in arrays
                size_t end_quote = content.find('"', pos + 1);
                if (end_quote != std::string::npos) {
                    current_element += content.substr(pos, end_quote - pos + 1);
                    pos = end_quote;
                } else {
                    current_element += c;
                }
            } else {
                current_element += c;
            }
            
            ++pos;
        }
        
        // Don't forget the last element
        std::string elem = trim(current_element);
        if (!elem.empty()) {
            result.push_back(parse_value(elem));
        }
        
        return result;
    }
    
    static void write_value(std::ofstream& file, const TOMLValue& value) {
        if (value.is_string()) {
            // Escape special characters
            std::string escaped = value.as_string();
            size_t pos = 0;
            while ((pos = escaped.find('"', pos)) != std::string::npos) {
                escaped.replace(pos, 1, "\\\"");
                pos += 2;
            }
            pos = 0;
            while ((pos = escaped.find('\n', pos)) != std::string::npos) {
                escaped.replace(pos, 1, "\\n");
                pos += 2;
            }
            file << "\"" << escaped << "\"";
        } else if (value.is_int()) {
            file << value.as_int();
        } else if (value.is_float()) {
            // Ensure sufficient precision
            file << std::setprecision(15) << value.as_float();
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
        }
    }
};

// ============================================================================
// TOML Configuration Loading/Saving
// ============================================================================
bool ConfigManager::load_from_file(const Path& config_path) {
    TOMLParser::TOMLDocument doc = TOMLParser::parse(config_path.string());
    
    if (doc.empty()) {
        std::cerr << "Warning: Could not parse config file: " << config_path << std::endl;
        return false;
    }
    
    // Parse all sections
    parse_memory_section_toml(doc);
    parse_performance_section_toml(doc);
    parse_logging_section_toml(doc);
    parse_power_section_toml(doc);
    parse_benchmark_section_toml(doc);
    parse_codec_section_toml(doc);
    parse_model_section_toml(doc);
    
    return validate_config();
}

bool ConfigManager::save_to_file(const Path& config_path) {
    TOMLParser::TOMLDocument doc;
    
    // Memory section
    doc["memory"]["vram_budget_gb"] = TOMLParser::TOMLValue{static_cast<int64_t>(memory_config_.vram_budget_gb)};
    doc["memory"]["ram_budget_gb"] = TOMLParser::TOMLValue{static_cast<int64_t>(memory_config_.ram_budget_gb)};
    doc["memory"]["prefetch_lookahead"] = TOMLParser::TOMLValue{static_cast<int64_t>(memory_config_.prefetch_lookahead)};
    doc["memory"]["eviction_aggression"] = TOMLParser::TOMLValue{static_cast<double>(memory_config_.eviction_aggression)};
    doc["memory"]["enable_compression"] = TOMLParser::TOMLValue{memory_config_.enable_compression};
    doc["memory"]["compression_algorithm"] = TOMLParser::TOMLValue{memory_config_.compression_algorithm};
    doc["memory"]["max_packs_in_memory"] = TOMLParser::TOMLValue{static_cast<int64_t>(memory_config_.max_packs_in_memory)};
    
    // Performance section
    doc["performance"]["enable_gpu"] = TOMLParser::TOMLValue{perf_config_.enable_gpu};
    doc["performance"]["enable_profiling"] = TOMLParser::TOMLValue{perf_config_.enable_profiling};
    doc["performance"]["workgroup_size_x"] = TOMLParser::TOMLValue{static_cast<int64_t>(perf_config_.workgroup_size_x)};
    doc["performance"]["workgroup_size_y"] = TOMLParser::TOMLValue{static_cast<int64_t>(perf_config_.workgroup_size_y)};
    doc["performance"]["use_subgroup_ops"] = TOMLParser::TOMLValue{perf_config_.use_subgroup_ops};
    doc["performance"]["use_fp16_math"] = TOMLParser::TOMLValue{perf_config_.use_fp16_math};
    doc["performance"]["scale_factor"] = TOMLParser::TOMLValue{static_cast<double>(perf_config_.scale_factor)};
    
    // Logging section
    doc["logging"]["log_level"] = TOMLParser::TOMLValue{static_cast<int64_t>(static_cast<int>(logging_config_.log_level))};
    doc["logging"]["log_to_file"] = TOMLParser::TOMLValue{logging_config_.log_to_file};
    doc["logging"]["log_file_path"] = TOMLParser::TOMLValue{logging_config_.log_file_path};
    doc["logging"]["log_performance"] = TOMLParser::TOMLValue{logging_config_.log_performance};
    doc["logging"]["log_memory_usage"] = TOMLParser::TOMLValue{logging_config_.log_memory_usage};
    doc["logging"]["max_log_file_size_mb"] = TOMLParser::TOMLValue{static_cast<int64_t>(logging_config_.max_log_file_size_mb)};
    
    // Power section
    doc["power"]["enable_power_saver"] = TOMLParser::TOMLValue{power_config_.enable_power_saver};
    doc["power"]["auto_detect_battery"] = TOMLParser::TOMLValue{power_config_.auto_detect_battery};
    doc["power"]["power_profile"] = TOMLParser::TOMLValue{static_cast<int64_t>(power_config_.power_profile)};
    doc["power"]["battery_threshold_percent"] = TOMLParser::TOMLValue{static_cast<int64_t>(power_config_.battery_threshold_percent)};
    doc["power"]["throttle_on_thermal"] = TOMLParser::TOMLValue{power_config_.throttle_on_thermal};
    doc["power"]["max_workgroup_size_battery"] = TOMLParser::TOMLValue{static_cast<int64_t>(power_config_.max_workgroup_size_battery)};
    doc["power"]["prefetch_lookahead_battery"] = TOMLParser::TOMLValue{static_cast<int64_t>(power_config_.prefetch_lookahead_battery)};
    doc["power"]["disable_profiling_on_battery"] = TOMLParser::TOMLValue{power_config_.disable_profiling_on_battery};
    
    // Benchmark section
    doc["benchmark"]["enable_benchmark_mode"] = TOMLParser::TOMLValue{benchmark_config_.enable_benchmark_mode};
    doc["benchmark"]["warmup_tokens"] = TOMLParser::TOMLValue{static_cast<int64_t>(benchmark_config_.warmup_tokens)};
    doc["benchmark"]["benchmark_tokens"] = TOMLParser::TOMLValue{static_cast<int64_t>(benchmark_config_.benchmark_tokens)};
    doc["benchmark"]["iterations"] = TOMLParser::TOMLValue{static_cast<int64_t>(benchmark_config_.iterations)};
    doc["benchmark"]["output_json"] = TOMLParser::TOMLValue{benchmark_config_.output_json};
    doc["benchmark"]["output_file"] = TOMLParser::TOMLValue{benchmark_config_.output_file};
    doc["benchmark"]["test_power_modes"] = TOMLParser::TOMLValue{benchmark_config_.test_power_modes};
    doc["benchmark"]["test_memory_pressure"] = TOMLParser::TOMLValue{benchmark_config_.test_memory_pressure};
    
    // Codec section
    doc["codec"]["enable_compression"] = TOMLParser::TOMLValue{codec_config_.enable_compression};
    doc["codec"]["algorithm"] = TOMLParser::TOMLValue{codec_config_.algorithm};
    doc["codec"]["compression_level"] = TOMLParser::TOMLValue{static_cast<int64_t>(codec_config_.compression_level)};
    doc["codec"]["decompression_threads"] = TOMLParser::TOMLValue{static_cast<int64_t>(codec_config_.decompression_threads)};
    doc["codec"]["enable_blosc2"] = TOMLParser::TOMLValue{codec_config_.enable_blosc2};
    doc["codec"]["enable_zfp"] = TOMLParser::TOMLValue{codec_config_.enable_zfp};
    doc["codec"]["hybrid_compression_ratio"] = TOMLParser::TOMLValue{static_cast<double>(codec_config_.hybrid_compression_ratio)};
    
    // Model section
    doc["model"]["model_path"] = TOMLParser::TOMLValue{model_path_};
    
    bool success = TOMLParser::write(config_path.string(), doc);
    if (success) {
        std::cout << "Configuration saved to: " << config_path << std::endl;
    }
    return success;
}

void ConfigManager::parse_memory_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("memory");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_int()) {
            target = static_cast<decltype(target)>(kv->second.as_int());
        }
    };
    
    auto get_float = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_float()) {
            target = static_cast<decltype(target)>(kv->second.as_float());
        }
    };
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_bool()) {
            target = kv->second.as_bool();
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_string()) {
            target = kv->second.as_string();
        }
    };
    
    get_int("vram_budget_gb", memory_config_.vram_budget_gb);
    get_int("ram_budget_gb", memory_config_.ram_budget_gb);
    get_int("prefetch_lookahead", memory_config_.prefetch_lookahead);
    get_float("eviction_aggression", memory_config_.eviction_aggression);
    get_bool("enable_compression", memory_config_.enable_compression);
    get_string("compression_algorithm", memory_config_.compression_algorithm);
    get_int("max_packs_in_memory", memory_config_.max_packs_in_memory);
}

void ConfigManager::parse_performance_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("performance");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_bool()) {
            target = kv->second.as_bool();
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_int()) {
            target = static_cast<decltype(target)>(kv->second.as_int());
        }
    };
    
    auto get_float = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_float()) {
            target = static_cast<decltype(target)>(kv->second.as_float());
        }
    };
    
    get_bool("enable_gpu", perf_config_.enable_gpu);
    get_bool("enable_profiling", perf_config_.enable_profiling);
    get_int("workgroup_size_x", perf_config_.workgroup_size_x);
    get_int("workgroup_size_y", perf_config_.workgroup_size_y);
    get_bool("use_subgroup_ops", perf_config_.use_subgroup_ops);
    get_bool("use_fp16_math", perf_config_.use_fp16_math);
    get_float("scale_factor", perf_config_.scale_factor);
}

void ConfigManager::parse_logging_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("logging");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_int()) {
            target = static_cast<decltype(target)>(kv->second.as_int());
        }
    };
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_bool()) {
            target = kv->second.as_bool();
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_string()) {
            target = kv->second.as_string();
        }
    };
    
    get_int("log_level", logging_config_.log_level);
    get_bool("log_to_file", logging_config_.log_to_file);
    get_string("log_file_path", logging_config_.log_file_path);
    get_bool("log_performance", logging_config_.log_performance);
    get_bool("log_memory_usage", logging_config_.log_memory_usage);
    get_int("max_log_file_size_mb", logging_config_.max_log_file_size_mb);
}

void ConfigManager::parse_power_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("power");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_bool()) {
            target = kv->second.as_bool();
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_int()) {
            target = static_cast<decltype(target)>(kv->second.as_int());
        }
    };
    
    get_bool("enable_power_saver", power_config_.enable_power_saver);
    get_bool("auto_detect_battery", power_config_.auto_detect_battery);
    get_int("power_profile", power_config_.power_profile);
    get_int("battery_threshold_percent", power_config_.battery_threshold_percent);
    get_bool("throttle_on_thermal", power_config_.throttle_on_thermal);
    get_int("max_workgroup_size_battery", power_config_.max_workgroup_size_battery);
    get_int("prefetch_lookahead_battery", power_config_.prefetch_lookahead_battery);
    get_bool("disable_profiling_on_battery", power_config_.disable_profiling_on_battery);
}

void ConfigManager::parse_benchmark_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("benchmark");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_bool()) {
            target = kv->second.as_bool();
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_int()) {
            target = static_cast<decltype(target)>(kv->second.as_int());
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_string()) {
            target = kv->second.as_string();
        }
    };
    
    get_bool("enable_benchmark_mode", benchmark_config_.enable_benchmark_mode);
    get_int("warmup_tokens", benchmark_config_.warmup_tokens);
    get_int("benchmark_tokens", benchmark_config_.benchmark_tokens);
    get_int("iterations", benchmark_config_.iterations);
    get_bool("output_json", benchmark_config_.output_json);
    get_string("output_file", benchmark_config_.output_file);
    get_bool("test_power_modes", benchmark_config_.test_power_modes);
    get_bool("test_memory_pressure", benchmark_config_.test_memory_pressure);
}

void ConfigManager::parse_codec_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("codec");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_bool()) {
            target = kv->second.as_bool();
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_int()) {
            target = static_cast<decltype(target)>(kv->second.as_int());
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_string()) {
            target = kv->second.as_string();
        }
    };
    
    auto get_float = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end() && kv->second.is_float()) {
            target = static_cast<decltype(target)>(kv->second.as_float());
        }
    };
    
    get_bool("enable_compression", codec_config_.enable_compression);
    get_string("algorithm", codec_config_.algorithm);
    get_int("compression_level", codec_config_.compression_level);
    get_int("decompression_threads", codec_config_.decompression_threads);
    get_bool("enable_blosc2", codec_config_.enable_blosc2);
    get_bool("enable_zfp", codec_config_.enable_zfp);
    get_float("hybrid_compression_ratio", codec_config_.hybrid_compression_ratio);
}

void ConfigManager::parse_model_section_toml(const TOMLParser::TOMLDocument& doc) {
    auto it = doc.find("model");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto kv = table.find("model_path");
    if (kv != table.end() && kv->second.is_string()) {
        model_path_ = kv->second.as_string();
    }
}

void ConfigManager::load_from_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            model_path_ = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            load_from_file(argv[++i]);
        } else if (arg == "--vram-gb" && i + 1 < argc) {
            memory_config_.vram_budget_gb = std::stoull(argv[++i]);
        } else if (arg == "--ram-gb" && i + 1 < argc) {
            memory_config_.ram_budget_gb = std::stoull(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            logging_config_.log_level = LogLevel::DEBUG;
        } else if (arg == "--power-profile" && i + 1 < argc) {
            power_config_.power_profile = std::stoul(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark_config_.enable_benchmark_mode = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            // This would be passed to the engine
            (void)argv[++i];
        }
    }
}

void ConfigManager::set_defaults() {
    memory_config_ = {
        .vram_budget_gb = 4,
        .ram_budget_gb = 16,
        .prefetch_lookahead = 3,
        .eviction_aggression = 0.7f,
        .enable_compression = true,
        .compression_algorithm = "blosc2",
        .max_packs_in_memory = 64
    };
    
    perf_config_ = {
        .enable_gpu = true,
        .enable_profiling = false,
        .workgroup_size_x = 16,
        .workgroup_size_y = 16,
        .use_subgroup_ops = true,
        .use_fp16_math = true,
        .scale_factor = 1.0f
    };
    
    logging_config_ = {
        .log_level = LogLevel::INFO,
        .log_to_file = false,
        .log_file_path = "vk_symbiote.log",
        .log_performance = true,
        .log_memory_usage = true,
        .max_log_file_size_mb = 100
    };
    
    power_config_ = {
        .enable_power_saver = false,
        .auto_detect_battery = true,
        .power_profile = 1,  // balanced
        .battery_threshold_percent = 30,
        .throttle_on_thermal = true,
        .max_workgroup_size_battery = 64,
        .prefetch_lookahead_battery = 1,
        .disable_profiling_on_battery = true
    };
    
    benchmark_config_ = {
        .enable_benchmark_mode = false,
        .warmup_tokens = 10,
        .benchmark_tokens = 100,
        .iterations = 3,
        .output_json = false,
        .output_file = "benchmark_results.json",
        .test_power_modes = true,
        .test_memory_pressure = true
    };
    
    codec_config_ = {
        .enable_compression = true,
        .algorithm = "hybrid",
        .compression_level = 5,
        .decompression_threads = 4,
        .enable_blosc2 = true,
        .enable_zfp = true,
        .hybrid_compression_ratio = 0.5f
    };
    
    model_path_ = "";
}

void ConfigManager::parse_memory_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    (void)config; // Legacy method, now handled by TOML parser
}

void ConfigManager::parse_performance_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    (void)config;
}

void ConfigManager::parse_logging_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    (void)config;
}

void ConfigManager::parse_power_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    (void)config;
}

void ConfigManager::parse_benchmark_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    (void)config;
}

bool ConfigManager::validate_config() const {
    bool valid = true;
    
    if (memory_config_.vram_budget_gb == 0 || memory_config_.vram_budget_gb > 64) {
        std::cerr << "Error: VRAM budget must be between 1-64 GB" << std::endl;
        valid = false;
    }
    
    if (memory_config_.ram_budget_gb == 0 || memory_config_.ram_budget_gb > 128) {
        std::cerr << "Error: RAM budget must be between 1-128 GB" << std::endl;
        valid = false;
    }
    
    if (memory_config_.prefetch_lookahead == 0 || memory_config_.prefetch_lookahead > 10) {
        std::cerr << "Error: Prefetch lookahead must be between 1-10" << std::endl;
        valid = false;
    }
    
    if (memory_config_.eviction_aggression < 0.0f || memory_config_.eviction_aggression > 1.0f) {
        std::cerr << "Error: Eviction aggression must be between 0.0-1.0" << std::endl;
        valid = false;
    }
    
    return valid;
}

void ConfigManager::print_config() const {
    std::cout << "=== Vulkan Symbiote Configuration ===" << std::endl;
    std::cout << "Memory:" << std::endl;
    std::cout << "  VRAM Budget: " << memory_config_.vram_budget_gb << " GB" << std::endl;
    std::cout << "  RAM Budget: " << memory_config_.ram_budget_gb << " GB" << std::endl;
    std::cout << "  Prefetch Lookahead: " << memory_config_.prefetch_lookahead << std::endl;
    std::cout << "  Eviction Aggression: " << memory_config_.eviction_aggression << std::endl;
    std::cout << "  Compression: " << (memory_config_.enable_compression ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Algorithm: " << memory_config_.compression_algorithm << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  GPU Enabled: " << (perf_config_.enable_gpu ? "Yes" : "No") << std::endl;
    std::cout << "  FP16 Math: " << (perf_config_.use_fp16_math ? "Yes" : "No") << std::endl;
    std::cout << "  Workgroup Size: " << perf_config_.workgroup_size_x << "x" << perf_config_.workgroup_size_y << std::endl;
    std::cout << "  Subgroup Ops: " << (perf_config_.use_subgroup_ops ? "Yes" : "No") << std::endl;
    std::cout << "Power:" << std::endl;
    std::cout << "  Power Saver: " << (power_config_.enable_power_saver ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Profile: " << (power_config_.power_profile == 0 ? "High Performance" : 
                                   power_config_.power_profile == 2 ? "Power Saver" : "Balanced") << std::endl;
    std::cout << "  Auto Battery Detect: " << (power_config_.auto_detect_battery ? "Yes" : "No") << std::endl;
    std::cout << "Benchmark:" << std::endl;
    std::cout << "  Mode: " << (benchmark_config_.enable_benchmark_mode ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Tokens: " << benchmark_config_.warmup_tokens << " warmup + " << benchmark_config_.benchmark_tokens << std::endl;
    std::cout << "  Iterations: " << benchmark_config_.iterations << std::endl;
    std::cout << "Codec:" << std::endl;
    std::cout << "  Algorithm: " << codec_config_.algorithm << std::endl;
    std::cout << "  Blosc2: " << (codec_config_.enable_blosc2 ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  ZFP: " << (codec_config_.enable_zfp ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Model Path: " << (model_path_.empty() ? "(Not set)" : model_path_) << std::endl;
    std::cout << "=====================================" << std::endl;
}

} // namespace vk_symbiote
