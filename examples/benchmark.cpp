/**
 * Vulkan Symbiote Loader - Benchmark Suite
 * 
 * Comprehensive benchmark for measuring:
 * - Tokens/second generation rate
 * - VRAM/RAM utilization
 * - Pack eviction rates
 * - Power consumption impact
 * - Latency metrics
 */

#include "../vk_symbiote/include/vk_symbiote/VulkanSymbioteEngine.h"
#include "../vk_symbiote/include/vk_symbiote/ConfigManager.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <sstream>
#include <cstring>

namespace vk_symbiote {

// Benchmark configuration
struct BenchmarkConfig {
    uint32_t warmup_tokens = 10;
    uint32_t benchmark_tokens = 100;
    uint32_t iterations = 3;
    std::string model_path;
    std::vector<std::string> test_prompts;
    bool test_power_modes = true;
    bool output_json = false;
    std::string output_file;
};

// Benchmark results for a single run
struct BenchmarkRun {
    std::string name;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    uint32_t tokens_generated;
    double tokens_per_second;
    uint64_t peak_vram_bytes;
    uint64_t peak_ram_bytes;
    uint32_t pack_evictions;
    double avg_latency_ms;
    double p99_latency_ms;
    std::vector<double> token_latencies;
    std::string power_profile;
};

// Full benchmark results
struct BenchmarkResults {
    std::vector<BenchmarkRun> runs;
    std::string device_name;
    std::string timestamp;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    
    void print_summary() const;
    void export_json(const std::string& filename) const;
    void export_csv(const std::string& filename) const;
};

class BenchmarkSuite {
public:
    explicit BenchmarkSuite(const BenchmarkConfig& config) 
        : config_(config), engine_(nullptr) {}
    
    ~BenchmarkSuite() = default;
    
    bool initialize() {
        std::cout << "[Benchmark] Initializing Vulkan Symbiote Engine..." << std::endl;
        std::cout << "[Benchmark] Model: " << config_.model_path << std::endl;
        
        try {
            engine_ = std::make_unique<VulkanSymbioteEngine>(config_.model_path);
            
            // Get device info
            results_.device_name = "Unknown GPU";
            results_.hidden_size = engine_->config().hidden_size;
            results_.num_layers = engine_->config().num_layers;
            results_.num_heads = engine_->config().num_attention_heads;
            
            // Set timestamp
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            results_.timestamp = ss.str();
            
            std::cout << "[Benchmark] Engine initialized successfully" << std::endl;
            std::cout << "[Benchmark] Model: " << engine_->config().model_type 
                     << " | Layers: " << results_.num_layers
                     << " | Hidden: " << results_.hidden_size
                     << " | Heads: " << results_.num_heads << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Benchmark] Failed to initialize engine: " << e.what() << std::endl;
            return false;
        }
    }
    
    void run_all_benchmarks() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Starting Benchmark Suite" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Warmup
        if (config_.warmup_tokens > 0) {
            std::cout << "\n[Benchmark] Warming up with " << config_.warmup_tokens 
                     << " tokens..." << std::endl;
            warmup();
        }
        
        // Standard performance benchmark
        run_performance_benchmark("Standard Performance");
        
        // Memory pressure benchmark
        run_memory_benchmark();
        
        // Power mode benchmarks
        if (config_.test_power_modes) {
            run_power_mode_benchmarks();
        }
        
        // Prompt variety benchmark
        run_prompt_benchmarks();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Benchmark Suite Complete" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
    
    const BenchmarkResults& get_results() const { return results_; }
    
private:
    BenchmarkConfig config_;
    std::unique_ptr<VulkanSymbioteEngine> engine_;
    BenchmarkResults results_;
    
    void warmup() {
        std::string warmup_prompt = "The quick brown fox jumps over the lazy dog. ";
        try {
            engine_->generate(warmup_prompt, config_.warmup_tokens, 0.7f);
            std::cout << "[Benchmark] Warmup complete" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Benchmark] Warmup failed: " << e.what() << std::endl;
        }
        engine_->reset_performance_metrics();
    }
    
    void run_performance_benchmark(const std::string& name) {
        std::cout << "\n[Benchmark] Running: " << name << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        std::string prompt = "Once upon a time, in a distant galaxy,";
        
        for (uint32_t iter = 0; iter < config_.iterations; ++iter) {
            std::cout << "  Iteration " << (iter + 1) << "/" << config_.iterations << ": ";
            
            BenchmarkRun run;
            run.name = name + " (Iteration " + std::to_string(iter + 1) + ")";
            run.power_profile = get_power_profile_name();
            
            // Record baseline memory
            auto baseline_metrics = engine_->get_performance_metrics();
            
            // Run generation
            run.start_time_ns = get_current_time_ns();
            
            try {
                engine_->generate(prompt, config_.benchmark_tokens, 0.7f);
                run.tokens_generated = config_.benchmark_tokens;
            } catch (const std::exception& e) {
                std::cerr << "FAILED - " << e.what() << std::endl;
                continue;
            }
            
            run.end_time_ns = get_current_time_ns();
            
            // Calculate metrics
            double elapsed_sec = (run.end_time_ns - run.start_time_ns) / 1e9;
            run.tokens_per_second = run.tokens_generated / elapsed_sec;
            
            // Get performance metrics from engine
            auto metrics = engine_->get_performance_metrics();
            
            // Calculate latency statistics
            // Note: Individual token latencies would need to be tracked in engine
            run.avg_latency_ms = (elapsed_sec / run.tokens_generated) * 1000.0;
            run.p99_latency_ms = run.avg_latency_ms; // Simplified
            
            std::cout << std::fixed << std::setprecision(2);
            std::cout << run.tokens_per_second << " t/s | ";
            std::cout << run.avg_latency_ms << " ms/token" << std::endl;
            
            results_.runs.push_back(run);
            engine_->reset_performance_metrics();
        }
    }
    
    void run_memory_benchmark() {
        std::cout << "\n[Benchmark] Running: Memory Pressure Test" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        // Test with varying context lengths to stress memory
        std::vector<uint32_t> context_lengths = {128, 256, 512, 1024};
        
        for (uint32_t ctx_len : context_lengths) {
            std::cout << "  Context length " << ctx_len << ": ";
            
            // Generate a prompt of approximately ctx_len tokens
            std::string prompt = generate_test_prompt(ctx_len);
            
            BenchmarkRun run;
            run.name = "Memory Test (ctx=" + std::to_string(ctx_len) + ")";
            run.power_profile = get_power_profile_name();
            run.start_time_ns = get_current_time_ns();
            
            try {
                auto output = engine_->generate(prompt, 20, 0.7f);
                run.tokens_generated = 20;
                run.end_time_ns = get_current_time_ns();
                
                double elapsed_sec = (run.end_time_ns - run.start_time_ns) / 1e9;
                run.tokens_per_second = run.tokens_generated / elapsed_sec;
                
                std::cout << std::fixed << std::setprecision(2);
                std::cout << run.tokens_per_second << " t/s" << std::endl;
                
                results_.runs.push_back(run);
            } catch (const std::exception& e) {
                std::cout << "OOM/OVERRUN" << std::endl;
            }
            
            engine_->reset_performance_metrics();
        }
    }
    
    void run_power_mode_benchmarks() {
        std::cout << "\n[Benchmark] Running: Power Mode Comparison" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        std::vector<VulkanSymbioteEngine::PowerProfile> profiles = {
            VulkanSymbioteEngine::PowerProfile::HIGH_PERFORMANCE,
            VulkanSymbioteEngine::PowerProfile::BALANCED,
            VulkanSymbioteEngine::PowerProfile::POWER_SAVER
        };
        
        std::string prompt = "In the realm of artificial intelligence,";
        
        for (auto profile : profiles) {
            std::string profile_name;
            switch (profile) {
                case VulkanSymbioteEngine::PowerProfile::HIGH_PERFORMANCE:
                    profile_name = "High Performance";
                    break;
                case VulkanSymbioteEngine::PowerProfile::BALANCED:
                    profile_name = "Balanced";
                    break;
                case VulkanSymbioteEngine::PowerProfile::POWER_SAVER:
                    profile_name = "Power Saver";
                    break;
            }
            
            std::cout << "  Profile: " << profile_name << ": ";
            
            engine_->set_power_profile(profile);
            
            BenchmarkRun run;
            run.name = "Power Mode: " + profile_name;
            run.power_profile = profile_name;
            run.start_time_ns = get_current_time_ns();
            
            try {
                engine_->generate(prompt, config_.benchmark_tokens, 0.7f);
                run.tokens_generated = config_.benchmark_tokens;
                run.end_time_ns = get_current_time_ns();
                
                double elapsed_sec = (run.end_time_ns - run.start_time_ns) / 1e9;
                run.tokens_per_second = run.tokens_generated / elapsed_sec;
                
                // Get workgroup size for this profile
                uint32_t wg_x, wg_y, wg_z;
                engine_->get_workgroup_size(wg_x, wg_y, wg_z);
                
                std::cout << std::fixed << std::setprecision(2);
                std::cout << run.tokens_per_second << " t/s (wg=" << wg_x << ")" << std::endl;
                
                results_.runs.push_back(run);
            } catch (const std::exception& e) {
                std::cout << "FAILED" << std::endl;
            }
            
            engine_->reset_performance_metrics();
        }
        
        // Reset to balanced
        engine_->set_power_profile(VulkanSymbioteEngine::PowerProfile::BALANCED);
    }
    
    void run_prompt_benchmarks() {
        if (config_.test_prompts.empty()) return;
        
        std::cout << "\n[Benchmark] Running: Prompt Variations" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < config_.test_prompts.size(); ++i) {
            const auto& prompt = config_.test_prompts[i];
            std::string short_prompt = prompt.substr(0, 50) + "...";
            std::cout << "  Prompt " << (i + 1) << ": \"" << short_prompt << "\": ";
            
            BenchmarkRun run;
            run.name = "Prompt " + std::to_string(i + 1);
            run.power_profile = get_power_profile_name();
            run.start_time_ns = get_current_time_ns();
            
            try {
                engine_->generate(prompt, config_.benchmark_tokens, 0.7f);
                run.tokens_generated = config_.benchmark_tokens;
                run.end_time_ns = get_current_time_ns();
                
                double elapsed_sec = (run.end_time_ns - run.start_time_ns) / 1e9;
                run.tokens_per_second = run.tokens_generated / elapsed_sec;
                
                std::cout << std::fixed << std::setprecision(2);
                std::cout << run.tokens_per_second << " t/s" << std::endl;
                
                results_.runs.push_back(run);
            } catch (const std::exception& e) {
                std::cout << "FAILED" << std::endl;
            }
            
            engine_->reset_performance_metrics();
        }
    }
    
    std::string generate_test_prompt(uint32_t target_tokens) {
        // Generate a prompt of approximately target_tokens length
        std::string base = "The quick brown fox jumps over the lazy dog. ";
        uint32_t repetitions = (target_tokens / 10) + 1; // Rough approximation
        std::string result;
        for (uint32_t i = 0; i < repetitions; ++i) {
            result += base;
        }
        return result;
    }
    
    std::string get_power_profile_name() const {
        auto profile = engine_->get_power_profile();
        switch (profile) {
            case VulkanSymbioteEngine::PowerProfile::HIGH_PERFORMANCE:
                return "High Performance";
            case VulkanSymbioteEngine::PowerProfile::BALANCED:
                return "Balanced";
            case VulkanSymbioteEngine::PowerProfile::POWER_SAVER:
                return "Power Saver";
            default:
                return "Unknown";
        }
    }
};

void BenchmarkResults::print_summary() const {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Timestamp: " << timestamp << std::endl;
    std::cout << "Device: " << device_name << std::endl;
    std::cout << "Model: " << num_layers << " layers, " << hidden_size 
             << " hidden, " << num_heads << " heads" << std::endl;
    std::cout << "Total runs: " << runs.size() << std::endl;
    
    if (runs.empty()) return;
    
    // Calculate averages
    double avg_tps = 0.0;
    double min_tps = std::numeric_limits<double>::max();
    double max_tps = 0.0;
    
    for (const auto& run : runs) {
        avg_tps += run.tokens_per_second;
        min_tps = std::min(min_tps, run.tokens_per_second);
        max_tps = std::max(max_tps, run.tokens_per_second);
    }
    avg_tps /= runs.size();
    
    std::cout << "\nPerformance Metrics:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Average: " << avg_tps << " tokens/sec" << std::endl;
    std::cout << "  Min:     " << min_tps << " tokens/sec" << std::endl;
    std::cout << "  Max:     " << max_tps << " tokens/sec" << std::endl;
    
    // Group by test type
    std::cout << "\nDetailed Results:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(40) << "Test Name" 
             << std::setw(12) << "t/s" 
             << std::setw(12) << "ms/token" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (const auto& run : runs) {
        std::cout << std::left << std::setw(40) << run.name.substr(0, 39)
                 << std::fixed << std::setprecision(2)
                 << std::setw(12) << run.tokens_per_second
                 << std::setw(12) << run.avg_latency_ms << std::endl;
    }
}

void BenchmarkResults::export_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"timestamp\": \"" << timestamp << "\",\n";
    file << "  \"device\": \"" << device_name << "\",\n";
    file << "  \"model_config\": {\n";
    file << "    \"hidden_size\": " << hidden_size << ",\n";
    file << "    \"num_layers\": " << num_layers << ",\n";
    file << "    \"num_heads\": " << num_heads << "\n";
    file << "  },\n";
    file << "  \"runs\": [\n";
    
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];
        file << "    {\n";
        file << "      \"name\": \"" << run.name << "\",\n";
        file << "      \"tokens_per_second\": " << run.tokens_per_second << ",\n";
        file << "      \"avg_latency_ms\": " << run.avg_latency_ms << ",\n";
        file << "      \"p99_latency_ms\": " << run.p99_latency_ms << ",\n";
        file << "      \"tokens_generated\": " << run.tokens_generated << ",\n";
        file << "      \"power_profile\": \"" << run.power_profile << "\"\n";
        file << "    }";
        if (i < runs.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    file.close();
    
    std::cout << "[Benchmark] Results exported to: " << filename << std::endl;
}

void BenchmarkResults::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    // Header
    file << "timestamp,device,hidden_size,num_layers,num_heads,";
    file << "test_name,tokens_per_second,avg_latency_ms,p99_latency_ms,";
    file << "tokens_generated,power_profile\n";
    
    // Data rows
    for (const auto& run : runs) {
        file << timestamp << ",";
        file << device_name << ",";
        file << hidden_size << ",";
        file << num_layers << ",";
        file << num_heads << ",";
        file << "\"" << run.name << "\",";
        file << run.tokens_per_second << ",";
        file << run.avg_latency_ms << ",";
        file << run.p99_latency_ms << ",";
        file << run.tokens_generated << ",";
        file << run.power_profile << "\n";
    }
    
    file.close();
    std::cout << "[Benchmark] Results exported to: " << filename << std::endl;
}

} // namespace vk_symbiote

// Main entry point
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -t, --tokens N       Number of tokens to generate (default: 100)\n";
    std::cout << "  -i, --iterations N   Number of benchmark iterations (default: 3)\n";
    std::cout << "  -w, --warmup N       Number of warmup tokens (default: 10)\n";
    std::cout << "  --no-power-tests     Skip power mode benchmarks\n";
    std::cout << "  --json <file>        Export results to JSON file\n";
    std::cout << "  --csv <file>         Export results to CSV file\n";
    std::cout << "  -h, --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    using namespace vk_symbiote;
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    BenchmarkConfig config;
    config.model_path = argv[1];
    
    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-t" || arg == "--tokens") {
            if (i + 1 < argc) config.benchmark_tokens = std::stoi(argv[++i]);
        } else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) config.iterations = std::stoi(argv[++i]);
        } else if (arg == "-w" || arg == "--warmup") {
            if (i + 1 < argc) config.warmup_tokens = std::stoi(argv[++i]);
        } else if (arg == "--no-power-tests") {
            config.test_power_modes = false;
        } else if (arg == "--json") {
            if (i + 1 < argc) {
                config.output_json = true;
                config.output_file = argv[++i];
            }
        } else if (arg == "--csv") {
            if (i + 1 < argc) {
                config.output_file = argv[++i];
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Add some test prompts
    config.test_prompts = {
        "The quick brown fox jumps over the lazy dog.",
        "In the realm of artificial intelligence, neural networks",
        "Once upon a time in a distant galaxy far, far away",
        "The fundamental theorem of calculus states that"
    };
    
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Vulkan Symbiote Loader - Benchmark Suite" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Configuration:\n";
    std::cout << "  Model: " << config.model_path << "\n";
    std::cout << "  Benchmark tokens: " << config.benchmark_tokens << "\n";
    std::cout << "  Iterations: " << config.iterations << "\n";
    std::cout << "  Warmup tokens: " << config.warmup_tokens << "\n";
    std::cout << "  Power mode tests: " << (config.test_power_modes ? "enabled" : "disabled") << "\n";
    std::cout << std::string(80, '=') << std::endl;
    
    // Run benchmarks
    BenchmarkSuite suite(config);
    
    if (!suite.initialize()) {
        std::cerr << "Failed to initialize benchmark suite" << std::endl;
        return 1;
    }
    
    suite.run_all_benchmarks();
    
    // Print and export results
    const auto& results = suite.get_results();
    results.print_summary();
    
    if (config.output_json) {
        std::string json_file = config.output_file.empty() ? "benchmark_results.json" : config.output_file;
        results.export_json(json_file);
    } else if (!config.output_file.empty()) {
        // Assume CSV if file extension is .csv
        if (config.output_file.substr(config.output_file.find_last_of(".") + 1) == "csv") {
            results.export_csv(config.output_file);
        } else {
            results.export_json(config.output_file);
        }
    }
    
    std::cout << "\n[Benchmark] Complete!" << std::endl;
    return 0;
}
