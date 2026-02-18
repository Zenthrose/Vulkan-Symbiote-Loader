#pragma once

#include "Common.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <memory>
#include <mutex>
#include <vector>

namespace vk_symbiote {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

struct MemoryConfig {
    float vram_budget_gb = 4.0f;
    float ram_budget_gb = 16.0f;
    uint32 prefetch_lookahead = 3;
    float eviction_aggression = 0.7f;
    bool enable_compression = true;
    std::string compression_algorithm = "hybrid";
    uint32 max_packs_in_memory = 1000;
    bool enable_defrag = true;
    uint32 defrag_interval_ms = 5000;
};

struct PerformanceConfig {
    bool enable_gpu = true;
    bool enable_profiling = false;
    uint32 workgroup_size_x = 256;
    uint32 workgroup_size_y = 1;
    uint32 workgroup_size_z = 1;
    bool use_subgroup_ops = true;
    uint32 subgroup_size = 32;
    bool use_fp16_math = true;
    float scale_factor = 1.0f;
    uint32 thread_pool_size = 0;  // 0 = auto
};

struct PowerConfig {
    bool enable_power_management = true;
    bool enable_power_saver = false;
    bool auto_detect_battery = true;
    uint32 power_profile = 1;  // 0=performance, 1=balanced, 2=powersaver
    float battery_threshold_low = 0.25f;
    float battery_threshold_critical = 0.10f;
    bool throttle_on_thermal = true;
    bool reduce_workgroup_on_battery = true;
    uint32 min_workgroup_size_battery = 64;
    uint32 prefetch_reduction_factor = 2;
    bool disable_profiling_on_battery = true;
    uint32 battery_threshold_percent = 30;
    uint32 max_workgroup_size_battery = 64;
    uint32 prefetch_lookahead_battery = 1;
};

struct BenchmarkConfig {
    bool enable_benchmark_mode = false;
    uint32 warmup_tokens = 10;
    uint32 benchmark_tokens = 100;
    uint32 iterations = 3;
    bool output_json = true;
    std::string output_file = "benchmark_results.json";
    bool test_power_modes = false;
    bool test_memory_pressure = false;
    bool detailed_layer_stats = false;
};

struct LoggingConfig {
    LogLevel log_level = LogLevel::INFO;
    bool log_to_file = false;
    std::string log_file_path = "vk_symbiote.log";
    bool log_performance = false;
    bool log_memory_usage = false;
    uint32 max_log_file_size_mb = 100;
    bool log_to_console = true;
};

struct CodecConfig {
    std::string codec = "hybrid";
    uint32 compression_level = 5;
    bool enable_blosc2 = true;
    std::string blosc2_compressor = "lz4";
    bool blosc2_shuffle = true;
    bool enable_zfp = false;
    uint32 zfp_precision = 16;
    float zfp_rate = 0.0f;
    bool enable_hybrid = true;
    std::string hybrid_mode = "auto";
    uint32 decompression_threads = 0;  // 0 = auto
    
    // Backward compatibility aliases
    bool enable_compression = true;
    std::string algorithm = "hybrid";
    float hybrid_compression_ratio = 0.5f;
};

struct BatchConfig {
    bool enable_batching = true;
    uint32 max_batch_size = 8;
    bool dynamic_batch_size = true;
    uint32 batch_timeout_ms = 100;
    bool prefetch_for_batch = true;
    bool share_kv_cache = false;
    uint32 max_sequence_length = 8192;
};

struct VitalityConfig {
    bool enabled = true;
    float learning_rate = 0.001f;
    float momentum = 0.9f;
    bool use_adam = false;
    std::string model_path = "vitality_model.toml";
};

struct ShaderConfig {
    std::string cache_dir = ".shader_cache";
    bool enable_cache = true;
    bool use_cooperative_matrix = true;
    uint32 coop_matrix_m = 16;
    uint32 coop_matrix_n = 16;
    uint32 coop_matrix_k = 16;
    bool auto_tune = true;
    std::string tuning_file = "shader_tuning.conf";
};

class ConfigManager {
public:
    static ConfigManager& instance();
    
    bool load_from_file(const Path& config_path);
    bool save_to_file(const Path& config_path);
    bool load_from_toml(const Path& config_path);
    bool save_to_toml(const Path& config_path);
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
    
    // Batch configuration
    const BatchConfig& batch() const { return batch_config_; }
    void set_batch(const BatchConfig& config) { batch_config_ = config; }
    
    // Vitality oracle configuration
    const VitalityConfig& vitality() const { return vitality_config_; }
    void set_vitality(const VitalityConfig& config) { vitality_config_ = config; }
    
    // Shader configuration
    const ShaderConfig& shader() const { return shader_config_; }
    void set_shader(const ShaderConfig& config) { shader_config_ = config; }
    
    // Model configuration
    const std::string& model_path() const { return model_path_; }
    void set_model_path(const std::string& path) { model_path_ = path; }
    const std::string& model_type() const { return model_type_; }
    void set_model_type(const std::string& type) { model_type_ = type; }
    
    // Getters for TOML config values
    int get_int(const std::string& section, const std::string& key, int default_value) const;
    float get_float(const std::string& section, const std::string& key, float default_value) const;
    bool get_bool(const std::string& section, const std::string& key, bool default_value) const;
    std::string get_string(const std::string& section, const std::string& key, const std::string& default_value) const;
    
    // Validation
    bool validate_config() const;
    void print_config() const;
    
    // Timestamp helper
    std::string get_current_timestamp() const;

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
    BatchConfig batch_config_;
    VitalityConfig vitality_config_;
    ShaderConfig shader_config_;
    std::string model_path_;
    std::string model_type_;
    
    static std::unique_ptr<ConfigManager> instance_;
    static std::mutex instance_mutex_;
    
    // Config data storage for INI-style parsing
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_data_;
    
    // TOML document storage (using variant for type safety)
    mutable std::unordered_map<std::string, std::unordered_map<std::string, 
        std::variant<int64_t, double, bool, std::string>>> toml_data_;
    
    // Helper methods for legacy INI format
    void parse_memory_section();
    void parse_performance_section();
    void parse_logging_section();
    void parse_power_section();
    void parse_benchmark_section();
    void parse_codec_section();
    void parse_batch_section();
    void parse_vitality_section();
    void parse_shader_section();
    void parse_model_section();
    
    // Helper methods for TOML parsing
    using TOMLDocument = std::unordered_map<std::string, 
        std::unordered_map<std::string, std::variant<int64_t, double, bool, std::string>>>;
    
    void parse_memory_section_toml(const TOMLDocument& doc);
    void parse_performance_section_toml(const TOMLDocument& doc);
    void parse_logging_section_toml(const TOMLDocument& doc);
    void parse_power_section_toml(const TOMLDocument& doc);
    void parse_benchmark_section_toml(const TOMLDocument& doc);
    void parse_codec_section_toml(const TOMLDocument& doc);
    void parse_batch_section_toml(const TOMLDocument& doc);
    void parse_vitality_section_toml(const TOMLDocument& doc);
    void parse_shader_section_toml(const TOMLDocument& doc);
    void parse_model_section_toml(const TOMLDocument& doc);
    
    // Value extraction helpers
    bool get_bool_value(const std::variant<int64_t, double, bool, std::string>& value);
    int64_t get_int_value(const std::variant<int64_t, double, bool, std::string>& value);
    double get_float_value(const std::variant<int64_t, double, bool, std::string>& value);
    std::string get_string_value(const std::variant<int64_t, double, bool, std::string>& value);
    
    // Legacy parsing helpers
    std::variant<int64_t, double, bool, std::string> parse_value(const std::string& value_str);
    
    // Legacy method signatures for backward compatibility
    void parse_memory_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_performance_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_logging_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_power_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
    void parse_benchmark_section(const std::unordered_map<std::string, std::variant<int, float, std::string>>& config);
};

} // namespace vk_symbiote
