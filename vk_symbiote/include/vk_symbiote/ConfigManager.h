#pragma once

#include "Common.h"
#include <string>
#include <unordered_map>
#include <variant>

namespace vk_symbiote {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

struct MemoryConfig {
    uint64 vram_budget_gb = 4;
    uint64 ram_budget_gb = 16;
    uint64 prefetch_lookahead = 3;
    float eviction_aggression = 0.7f;
    bool enable_compression = true;
    std::string compression_algorithm = "blosc2";
    uint32 max_packs_in_memory = 64;
};

struct PerformanceConfig {
    bool enable_gpu = true;
    bool enable_profiling = false;
    uint32 workgroup_size_x = 16;
    uint32 workgroup_size_y = 16;
    bool use_subgroup_ops = true;
    bool use_fp16_math = true;
    float scale_factor = 1.0f;
};

struct LoggingConfig {
    LogLevel log_level = LogLevel::INFO;
    bool log_to_file = false;
    std::string log_file_path = "vk_symbiote.log";
    bool log_performance = true;
    bool log_memory_usage = true;
    uint32 max_log_file_size_mb = 100;
};

class ConfigManager {
public:
    static ConfigManager& instance();
    
    bool load_from_file(const Path& config_path);
    bool save_to_file(const Path& config_path);
    void load_from_args(int argc, char* argv[]);
    void set_defaults();
    
    // Memory configuration
    const MemoryConfig& memory() const { return memory_config_; }
    void set_memory(const MemoryConfig& config) { memory_config_ = config; }
    
    // Performance configuration
    const PerformanceConfig& performance() const { return perf_config_; }
    void set_performance(const PerformanceConfig& config) { perf_config_ = config; }
    
    // Logging configuration
    const LoggingConfig& logging() const { return logging_config_; }
    void set_logging(const LoggingConfig& config) { logging_config_ = config; }
    
    // Model configuration
    const std::string& model_path() const { return model_path_; }
    void set_model_path(const std::string& path) { model_path_ = path; }
    
    // Validation
    bool validate_config() const;
    void print_config() const;

    // Allow make_unique to access private constructor
    struct PrivateTag {};
    ConfigManager(PrivateTag) {}

private:
    ConfigManager() = default;
    
    MemoryConfig memory_config_;
    PerformanceConfig perf_config_;
    LoggingConfig logging_config_;
    std::string model_path_;
    
    static std::unique_ptr<ConfigManager> instance_;
    static std::mutex instance_mutex_;
    
    // Helper methods
    void parse_memory_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_performance_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_logging_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
};

} // namespace vk_symbiote