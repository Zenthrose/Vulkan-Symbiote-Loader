#include "ConfigManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <variant>
#include <unordered_map>
#include <iomanip>

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
// Simple INI-style parser (TOML-compatible subset)
// ============================================================================
bool ConfigManager::load_from_file(const Path& config_path) {
    std::ifstream file(config_path.string());
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << config_path << std::endl;
        return false;
    }
    
    std::string line;
    std::string current_section;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        ++line_num;
        
        // Trim whitespace
        auto trim = [](const std::string& s) {
            size_t start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) return std::string("");
            size_t end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };
        
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Parse section header [section]
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }
        
        // Parse key-value pair
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string value_str = trim(line.substr(eq_pos + 1));
        
        if (current_section.empty()) continue;
        
        // Parse value
        std::variant<int64_t, double, bool, std::string> value;
        
        // Try boolean
        if (value_str == "true") {
            value = true;
        } else if (value_str == "false") {
            value = false;
        }
        // Try integer
        else if (value_str.find_first_not_of("0123456789-") == std::string::npos) {
            try {
                value = static_cast<int64_t>(std::stoll(value_str));
            } catch (...) {
                value = value_str;
            }
        }
        // Try float
        else if (value_str.find_first_of(".eE") != std::string::npos) {
            try {
                value = std::stod(value_str);
            } catch (...) {
                value = value_str;
            }
        }
        // String
        else {
            // Remove quotes if present
            if ((value_str.front() == '"' && value_str.back() == '"') ||
                (value_str.front() == '\'' && value_str.back() == '\'')) {
                value_str = value_str.substr(1, value_str.length() - 2);
            }
            value = value_str;
        }
        
        toml_data_[current_section][key] = value;
    }
    
    // Parse all sections
    parse_memory_section_toml(toml_data_);
    parse_performance_section_toml(toml_data_);
    parse_logging_section_toml(toml_data_);
    parse_power_section_toml(toml_data_);
    parse_benchmark_section_toml(toml_data_);
    parse_codec_section_toml(toml_data_);
    parse_model_section_toml(toml_data_);
    
    return validate_config();
}

bool ConfigManager::save_to_file(const Path& config_path) {
    std::ofstream file(config_path.string());
    if (!file.is_open()) {
        std::cerr << "Failed to open config file for writing: " << config_path << std::endl;
        return false;
    }
    
    file << "# Vulkan Symbiote Configuration\n";
    file << "# Auto-generated configuration file\n\n";
    
    // Memory section
    file << "[memory]\n";
    file << "vram_budget_gb = " << memory_config_.vram_budget_gb << "\n";
    file << "ram_budget_gb = " << memory_config_.ram_budget_gb << "\n";
    file << "prefetch_lookahead = " << memory_config_.prefetch_lookahead << "\n";
    file << "eviction_aggression = " << memory_config_.eviction_aggression << "\n";
    file << "enable_compression = " << (memory_config_.enable_compression ? "true" : "false") << "\n";
    file << "compression_algorithm = \"" << memory_config_.compression_algorithm << "\"\n";
    file << "max_packs_in_memory = " << memory_config_.max_packs_in_memory << "\n";
    file << "\n";
    
    // Performance section
    file << "[performance]\n";
    file << "enable_gpu = " << (perf_config_.enable_gpu ? "true" : "false") << "\n";
    file << "enable_profiling = " << (perf_config_.enable_profiling ? "true" : "false") << "\n";
    file << "workgroup_size_x = " << perf_config_.workgroup_size_x << "\n";
    file << "workgroup_size_y = " << perf_config_.workgroup_size_y << "\n";
    file << "use_subgroup_ops = " << (perf_config_.use_subgroup_ops ? "true" : "false") << "\n";
    file << "use_fp16_math = " << (perf_config_.use_fp16_math ? "true" : "false") << "\n";
    file << "scale_factor = " << perf_config_.scale_factor << "\n";
    file << "\n";
    
    // Logging section
    file << "[logging]\n";
    file << "log_level = " << static_cast<int>(logging_config_.log_level) << "\n";
    file << "log_to_file = " << (logging_config_.log_to_file ? "true" : "false") << "\n";
    file << "log_file_path = \"" << logging_config_.log_file_path << "\"\n";
    file << "log_performance = " << (logging_config_.log_performance ? "true" : "false") << "\n";
    file << "log_memory_usage = " << (logging_config_.log_memory_usage ? "true" : "false") << "\n";
    file << "max_log_file_size_mb = " << logging_config_.max_log_file_size_mb << "\n";
    file << "\n";
    
    // Power section
    file << "[power]\n";
    file << "enable_power_saver = " << (power_config_.enable_power_saver ? "true" : "false") << "\n";
    file << "auto_detect_battery = " << (power_config_.auto_detect_battery ? "true" : "false") << "\n";
    file << "power_profile = " << power_config_.power_profile << "\n";
    file << "battery_threshold_percent = " << power_config_.battery_threshold_percent << "\n";
    file << "throttle_on_thermal = " << (power_config_.throttle_on_thermal ? "true" : "false") << "\n";
    file << "max_workgroup_size_battery = " << power_config_.max_workgroup_size_battery << "\n";
    file << "prefetch_lookahead_battery = " << power_config_.prefetch_lookahead_battery << "\n";
    file << "disable_profiling_on_battery = " << (power_config_.disable_profiling_on_battery ? "true" : "false") << "\n";
    file << "\n";
    
    // Benchmark section
    file << "[benchmark]\n";
    file << "enable_benchmark_mode = " << (benchmark_config_.enable_benchmark_mode ? "true" : "false") << "\n";
    file << "warmup_tokens = " << benchmark_config_.warmup_tokens << "\n";
    file << "benchmark_tokens = " << benchmark_config_.benchmark_tokens << "\n";
    file << "iterations = " << benchmark_config_.iterations << "\n";
    file << "output_json = " << (benchmark_config_.output_json ? "true" : "false") << "\n";
    file << "output_file = \"" << benchmark_config_.output_file << "\"\n";
    file << "test_power_modes = " << (benchmark_config_.test_power_modes ? "true" : "false") << "\n";
    file << "test_memory_pressure = " << (benchmark_config_.test_memory_pressure ? "true" : "false") << "\n";
    file << "\n";
    
    // Codec section
    file << "[codec]\n";
    file << "enable_compression = " << (codec_config_.enable_compression ? "true" : "false") << "\n";
    file << "algorithm = \"" << codec_config_.algorithm << "\"\n";
    file << "compression_level = " << codec_config_.compression_level << "\n";
    file << "decompression_threads = " << codec_config_.decompression_threads << "\n";
    file << "enable_blosc2 = " << (codec_config_.enable_blosc2 ? "true" : "false") << "\n";
    file << "enable_zfp = " << (codec_config_.enable_zfp ? "true" : "false") << "\n";
    file << "hybrid_compression_ratio = " << codec_config_.hybrid_compression_ratio << "\n";
    file << "\n";
    
    // Model section
    file << "[model]\n";
    file << "model_path = \"" << model_path_ << "\"\n";
    
    std::cout << "Configuration saved to: " << config_path << std::endl;
    return file.good();
}

void ConfigManager::parse_memory_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("memory");
    if (it == doc.end()) return;
    
    const auto& table = it->second;

    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<int64_t>(kv->second));
            }
        }
    };

    auto get_float = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<double>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<double>(kv->second));
            }
        }
    };

    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                target = std::get<bool>(kv->second);
            }
        }
    };

    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<std::string>(kv->second)) {
                target = std::get<std::string>(kv->second);
            }
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

void ConfigManager::parse_performance_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("performance");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                target = std::get<bool>(kv->second);
            }
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<int64_t>(kv->second));
            }
        }
    };
    
    auto get_float = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<double>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<double>(kv->second));
            }
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

void ConfigManager::parse_logging_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("logging");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<int64_t>(kv->second));
            }
        }
    };
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                target = std::get<bool>(kv->second);
            }
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<std::string>(kv->second)) {
                target = std::get<std::string>(kv->second);
            }
        }
    };
    
    get_int("log_level", logging_config_.log_level);
    get_bool("log_to_file", logging_config_.log_to_file);
    get_string("log_file_path", logging_config_.log_file_path);
    get_bool("log_performance", logging_config_.log_performance);
    get_bool("log_memory_usage", logging_config_.log_memory_usage);
    get_int("max_log_file_size_mb", logging_config_.max_log_file_size_mb);
}

void ConfigManager::parse_power_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("power");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                target = std::get<bool>(kv->second);
            }
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<int64_t>(kv->second));
            }
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

void ConfigManager::parse_benchmark_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("benchmark");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                target = std::get<bool>(kv->second);
            }
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<int64_t>(kv->second));
            }
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<std::string>(kv->second)) {
                target = std::get<std::string>(kv->second);
            }
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

void ConfigManager::parse_codec_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("codec");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto get_bool = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                target = std::get<bool>(kv->second);
            }
        }
    };
    
    auto get_int = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<int64_t>(kv->second));
            }
        }
    };
    
    auto get_string = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<std::string>(kv->second)) {
                target = std::get<std::string>(kv->second);
            }
        }
    };
    
    auto get_float = [&](const std::string& key, auto& target) {
        auto kv = table.find(key);
        if (kv != table.end()) {
            if (std::holds_alternative<double>(kv->second)) {
                target = static_cast<std::remove_reference_t<decltype(target)>>(std::get<double>(kv->second));
            }
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

void ConfigManager::parse_model_section_toml(const std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>& doc) {
    auto it = doc.find("model");
    if (it == doc.end()) return;
    
    const auto& table = it->second;
    
    auto kv = table.find("model_path");
    if (kv != table.end()) {
        if (std::holds_alternative<std::string>(kv->second)) {
            model_path_ = std::get<std::string>(kv->second);
        }
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
        }
    }
}

void ConfigManager::set_defaults() {
    memory_config_ = MemoryConfig{};
    perf_config_ = PerformanceConfig{};
    logging_config_ = LoggingConfig{};
    power_config_ = PowerConfig{};
    benchmark_config_ = BenchmarkConfig{};
    codec_config_ = CodecConfig{};
    model_path_ = "";
}

void ConfigManager::parse_memory_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    (void)config;
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

int ConfigManager::get_int(const std::string& section, const std::string& key, int default_value) const {
    auto it = toml_data_.find(section);
    if (it != toml_data_.end()) {
        auto kv = it->second.find(key);
        if (kv != it->second.end()) {
            if (std::holds_alternative<int64_t>(kv->second)) {
                return static_cast<int>(std::get<int64_t>(kv->second));
            }
        }
    }
    return default_value;
}

float ConfigManager::get_float(const std::string& section, const std::string& key, float default_value) const {
    auto it = toml_data_.find(section);
    if (it != toml_data_.end()) {
        auto kv = it->second.find(key);
        if (kv != it->second.end()) {
            if (std::holds_alternative<double>(kv->second)) {
                return static_cast<float>(std::get<double>(kv->second));
            }
        }
    }
    return default_value;
}

bool ConfigManager::get_bool(const std::string& section, const std::string& key, bool default_value) const {
    auto it = toml_data_.find(section);
    if (it != toml_data_.end()) {
        auto kv = it->second.find(key);
        if (kv != it->second.end()) {
            if (std::holds_alternative<bool>(kv->second)) {
                return std::get<bool>(kv->second);
            }
        }
    }
    return default_value;
}

std::string ConfigManager::get_string(const std::string& section, const std::string& key, const std::string& default_value) const {
    auto it = toml_data_.find(section);
    if (it != toml_data_.end()) {
        auto kv = it->second.find(key);
        if (kv != it->second.end()) {
            if (std::holds_alternative<std::string>(kv->second)) {
                return std::get<std::string>(kv->second);
            }
        }
    }
    return default_value;
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
    
    return valid;
}

void ConfigManager::print_config() const {
    std::cout << "=== Vulkan Symbiote Configuration ===" << std::endl;
    std::cout << "Memory:" << std::endl;
    std::cout << "  VRAM Budget: " << memory_config_.vram_budget_gb << " GB" << std::endl;
    std::cout << "  RAM Budget: " << memory_config_.ram_budget_gb << " GB" << std::endl;
    std::cout << "  Prefetch Lookahead: " << memory_config_.prefetch_lookahead << std::endl;
    std::cout << "  Compression: " << (memory_config_.enable_compression ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  GPU Enabled: " << (perf_config_.enable_gpu ? "Yes" : "No") << std::endl;
    std::cout << "  FP16 Math: " << (perf_config_.use_fp16_math ? "Yes" : "No") << std::endl;
    std::cout << "Power:" << std::endl;
    std::cout << "  Profile: " << (power_config_.power_profile == 0 ? "High Performance" :
                                   power_config_.power_profile == 2 ? "Power Saver" : "Balanced") << std::endl;
    std::cout << "=====================================" << std::endl;
}

} // namespace vk_symbiote
