#include "ConfigManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

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

bool ConfigManager::load_from_file(const Path& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << config_path << std::endl;
        return false;
    }

    std::unordered_map<std::string, std::variant<int, float, std::string>> config;
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            std::transform(current_section.begin(), current_section.end(), 
                           current_section.begin(), ::tolower);
            continue;
        }
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value_str = line.substr(eq_pos + 1);
        
        key.erase(0, key.find_last_not_of(" \t"));
        value_str.erase(0, value_str.find_last_not_of(" \t"));
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        
        // Parse value
        try {
            if (value_str == "true") {
                config[key] = 1;
            } else if (value_str == "false") {
                config[key] = 0;
            } else {
                config[key] = value_str;
            }
        } catch (...) {
            config[key] = std::string("");
        }
    }
    
    // Parse sections
    if (!config.empty()) {
        if (config.find("memory") != config.end()) {
            parse_memory_section(config);
        }
        if (config.find("performance") != config.end()) {
            parse_performance_section(config);
        }
        if (config.find("logging") != config.end()) {
            parse_logging_section(config);
        }
        if (config.find("model") != config.end()) {
            auto it = config.find("model_path");
            if (it != config.end()) {
                model_path_ = std::get<std::string>(it->second);
            }
        }
    }
    
    return validate_config();
}

bool ConfigManager::save_to_file(const Path& config_path) {
    std::ofstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write config file: " << config_path << std::endl;
        return false;
    }
    
    file << "[memory]" << std::endl;
    file << "vram_budget_gb=" << memory_config_.vram_budget_gb << std::endl;
    file << "ram_budget_gb=" << memory_config_.ram_budget_gb << std::endl;
    file << "prefetch_lookahead=" << memory_config_.prefetch_lookahead << std::endl;
    file << "eviction_aggression=" << memory_config_.eviction_aggression << std::endl;
    file << "enable_compression=" << (memory_config_.enable_compression ? "true" : "false") << std::endl;
    file << "compression_algorithm=" << memory_config_.compression_algorithm << std::endl;
    file << "max_packs_in_memory=" << memory_config_.max_packs_in_memory << std::endl;
    
    file << "[performance]" << std::endl;
    file << "enable_gpu=" << (perf_config_.enable_gpu ? "true" : "false") << std::endl;
    file << "enable_profiling=" << (perf_config_.enable_profiling ? "true" : "false") << std::endl;
    file << "workgroup_size_x=" << perf_config_.workgroup_size_x << std::endl;
    file << "workgroup_size_y=" << perf_config_.workgroup_size_y << std::endl;
    file << "use_subgroup_ops=" << (perf_config_.use_subgroup_ops ? "true" : "false") << std::endl;
    file << "use_fp16_math=" << (perf_config_.use_fp16_math ? "true" : "false") << std::endl;
    file << "scale_factor=" << perf_config_.scale_factor << std::endl;
    
    file << "[logging]" << std::endl;
    file << "log_level=" << static_cast<int>(logging_config_.log_level) << std::endl;
    file << "log_to_file=" << (logging_config_.log_to_file ? "true" : "false") << std::endl;
    file << "log_file_path=" << logging_config_.log_file_path << std::endl;
    file << "log_performance=" << (logging_config_.log_performance ? "true" : "false") << std::endl;
    file << "log_memory_usage=" << (logging_config_.log_memory_usage ? "true" : "false") << std::endl;
    file << "max_log_file_size_mb=" << logging_config_.max_log_file_size_mb << std::endl;
    
    file << "[model]" << std::endl;
    file << "model_path=" << model_path_ << std::endl;
    
    file.close();
    return true;
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
    
    model_path_ = "";
}

void ConfigManager::parse_memory_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    auto it = config.find("vram_budget_gb");
    if (it != config.end()) {
        memory_config_.vram_budget_gb = static_cast<uint64>(std::get<int>(it->second));
    }
    it = config.find("ram_budget_gb");
    if (it != config.end()) {
        memory_config_.ram_budget_gb = static_cast<uint64>(std::get<int>(it->second));
    }
    it = config.find("prefetch_lookahead");
    if (it != config.end()) {
        memory_config_.prefetch_lookahead = static_cast<uint64>(std::get<int>(it->second));
    }
    it = config.find("eviction_aggression");
    if (it != config.end()) {
        memory_config_.eviction_aggression = std::get<float>(it->second);
    }
    it = config.find("enable_compression");
    if (it != config.end()) {
        memory_config_.enable_compression = std::get<int>(it->second) != 0;
    }
    it = config.find("compression_algorithm");
    if (it != config.end()) {
        memory_config_.compression_algorithm = std::get<std::string>(it->second);
    }
    it = config.find("max_packs_in_memory");
    if (it != config.end()) {
        memory_config_.max_packs_in_memory = static_cast<uint32>(std::get<int>(it->second));
    }
}

void ConfigManager::parse_performance_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    auto it = config.find("enable_gpu");
    if (it != config.end()) {
        perf_config_.enable_gpu = std::get<int>(it->second) != 0;
    }
    it = config.find("enable_profiling");
    if (it != config.end()) {
        perf_config_.enable_profiling = std::get<int>(it->second) != 0;
    }
    it = config.find("workgroup_size_x");
    if (it != config.end()) {
        perf_config_.workgroup_size_x = static_cast<uint32>(std::get<int>(it->second));
    }
    it = config.find("workgroup_size_y");
    if (it != config.end()) {
        perf_config_.workgroup_size_y = static_cast<uint32>(std::get<int>(it->second));
    }
    it = config.find("use_subgroup_ops");
    if (it != config.end()) {
        perf_config_.use_subgroup_ops = std::get<int>(it->second) != 0;
    }
    it = config.find("use_fp16_math");
    if (it != config.end()) {
        perf_config_.use_fp16_math = std::get<int>(it->second) != 0;
    }
    it = config.find("scale_factor");
    if (it != config.end()) {
        perf_config_.scale_factor = std::get<float>(it->second);
    }
}

void ConfigManager::parse_logging_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config) {
    auto it = config.find("log_level");
    if (it != config.end()) {
        logging_config_.log_level = static_cast<LogLevel>(std::get<int>(it->second));
    }
    it = config.find("log_to_file");
    if (it != config.end()) {
        logging_config_.log_to_file = std::get<int>(it->second) != 0;
    }
    it = config.find("log_file_path");
    if (it != config.end()) {
        logging_config_.log_file_path = std::get<std::string>(it->second);
    }
    it = config.find("log_performance");
    if (it != config.end()) {
        logging_config_.log_performance = std::get<int>(it->second) != 0;
    }
    it = config.find("log_memory_usage");
    if (it != config.end()) {
        logging_config_.log_memory_usage = std::get<int>(it->second) != 0;
    }
    it = config.find("max_log_file_size_mb");
    if (it != config.end()) {
        logging_config_.max_log_file_size_mb = std::get<uint32>(it->second);
    }
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
    std::cout << "Performance:" << std::endl;
    std::cout << "  GPU Enabled: " << (perf_config_.enable_gpu ? "Yes" : "No") << std::endl;
    std::cout << "  FP16 Math: " << (perf_config_.use_fp16_math ? "Yes" : "No") << std::endl;
    std::cout << "  Workgroup Size: " << perf_config_.workgroup_size_x << "x" << perf_config_.workgroup_size_y << std::endl;
    std::cout << "Model Path: " << (model_path_.empty() ? "(Not set)" : model_path_) << std::endl;
    std::cout << "=====================================" << std::endl;
}

} // namespace vk_symbiote