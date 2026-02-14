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

struct PowerConfig {
    bool enable_power_saver = false;
    bool auto_detect_battery = true;
    uint32 power_profile = 1;  // 0=high_perf, 1=balanced, 2=power_saver
    uint32 battery_threshold_percent = 30;
    bool throttle_on_thermal = true;
    uint32 max_workgroup_size_battery = 64;
    uint32 prefetch_lookahead_battery = 1;
    bool disable_profiling_on_battery = true;
};

struct BenchmarkConfig {
    bool enable_benchmark_mode = false;
    uint32 warmup_tokens = 10;
    uint32 benchmark_tokens = 100;
    uint32 iterations = 3;
    bool output_json = false;
    std::string output_file = "benchmark_results.json";
    bool test_power_modes = true;
    bool test_memory_pressure = true;
};

struct LoggingConfig {
    LogLevel log_level = LogLevel::INFO;
    bool log_to_file = false;
    std::string log_file_path = "vk_symbiote.log";
    bool log_performance = true;
    bool log_memory_usage = true;
    uint32 max_log_file_size_mb = 100;
};

struct CodecConfig {
    bool enable_compression = true;
    std::string algorithm = "hybrid";  // blosc2, zfp, hybrid
    uint32 compression_level = 5;
    uint32 decompression_threads = 4;
    bool enable_blosc2 = true;
    bool enable_zfp = true;
    float hybrid_compression_ratio = 0.5f;
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
    
    // Power management configuration
    const PowerConfig& power() const { return power_config_; }
    void set_power(const PowerConfig& config) { power_config_ = config; }
    
    // Benchmark configuration
    const BenchmarkConfig& benchmark() const { return benchmark_config_; }
    void set_benchmark(const BenchmarkConfig& config) { benchmark_config_ = config; }
    
    // Codec configuration
    const CodecConfig& codec() const { return codec_config_; }
    void set_codec(const CodecConfig& config) { codec_config_ = config; }
    
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
    PowerConfig power_config_;
    BenchmarkConfig benchmark_config_;
    CodecConfig codec_config_;
    std::string model_path_;
    
    static std::unique_ptr<ConfigManager> instance_;
    static std::mutex instance_mutex_;
    
    // Helper methods for legacy format
    void parse_memory_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_performance_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_logging_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_power_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_benchmark_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    
    // Forward declaration for TOML parser
    struct TOMLParser;
    
    // Helper methods for TOML parsing
    void parse_memory_section_toml(const TOMLParser::TOMLDocument& doc);
    void parse_performance_section_toml(const TOMLParser::TOMLDocument& doc);
    void parse_logging_section_toml(const TOMLParser::TOMLDocument& doc);
    void parse_power_section_toml(const TOMLParser::TOMLDocument& doc);
    void parse_benchmark_section_toml(const TOMLParser::TOMLDocument& doc);
    void parse_codec_section_toml(const TOMLParser::TOMLDocument& doc);
    void parse_model_section_toml(const TOMLParser::TOMLDocument& doc);
};

} // namespace vk_symbiote