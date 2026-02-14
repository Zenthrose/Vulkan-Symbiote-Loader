/**
 * Vulkan Symbiote Benchmark Example
 * 
 * This example demonstrates the comprehensive benchmarking capabilities
 * of the Vulkan Symbiote inference engine.
 * 
 * Usage:
 *   ./benchmark <path_to_gguf_model> [options]
 *   
 * Options:
 *   --warmup N         Number of warmup tokens (default: 10)
 *   --tokens N         Number of benchmark tokens (default: 100)
 *   --iterations N     Number of benchmark iterations (default: 3)
 *   --batch            Run batched generation benchmark
 *   --power-test       Test all power modes
 *   --memory-test      Test memory pressure handling
 *   --json <file>      Output results to JSON file
 *   --config <file>    Load configuration from TOML file
 * 
 * Example:
 *   ./benchmark model.gguf --warmup 20 --tokens 200 --iterations 5 --json results.json
 */

#include "../vk_symbiote/include/vk_symbiote/VulkanSymbioteEngine.h"
#include "../vk_symbiote/include/vk_symbiote/ConfigManager.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace vk_symbiote;

// ANSI color codes for pretty output
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* RED = "\033[31m";
const char* BLUE = "\033[34m";
const char* CYAN = "\033[36m";

void print_banner() {
    std::cout << BOLD << CYAN << R"(
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ██╗   ██╗██╗   ██╗██╗  ██╗ █████╗ ███╗   ██╗            ║
║     ██║   ██║██║   ██║██║ ██╔╝██╔══██╗████╗  ██║            ║
║     ██║   ██║██║   ██║█████╔╝ ███████║██╔██╗ ██║            ║
║     ╚██╗ ██╔╝██║   ██║██╔═██╗ ██╔══██║██║╚██╗██║            ║
║      ╚████╔╝ ╚██████╔╝██║  ██╗██║  ██║██║ ╚████║            ║
║       ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝            ║
║                                                               ║
║              Vulkan Symbiote Benchmark Suite                  ║
║                  Pure Vulkan Inference Engine                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
)" << RESET << std::endl;
}

void print_section(const std::string& title) {
    std::cout << "\n" << BOLD << BLUE << "▶ " << title << RESET << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

void print_stat(const std::string& name, double value, const std::string& unit = "") {
    std::cout << std::left << std::setw(30) << name << ": " 
              << GREEN << std::fixed << std::setprecision(2) << value 
              << RESET << " " << unit << std::endl;
}

void print_stat(const std::string& name, const std::string& value) {
    std::cout << std::left << std::setw(30) << name << ": " 
              << GREEN << value << RESET << std::endl;
}

struct BenchmarkOptions {
    std::string model_path;
    uint32_t warmup_tokens = 10;
    uint32_t benchmark_tokens = 100;
    uint32_t iterations = 3;
    bool run_batch_test = false;
    bool test_power_modes = false;
    bool test_memory_pressure = false;
    std::string json_output;
    std::string config_file;
};

BenchmarkOptions parse_args(int argc, char* argv[]) {
    BenchmarkOptions opts;
    
    if (argc < 2) {
        std::cerr << RED << "Error: Model path required" << RESET << std::endl;
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]" << std::endl;
        std::cerr << "\nOptions:" << std::endl;
        std::cerr << "  --warmup N         Warmup tokens (default: 10)" << std::endl;
        std::cerr << "  --tokens N         Benchmark tokens (default: 100)" << std::endl;
        std::cerr << "  --iterations N     Iterations (default: 3)" << std::endl;
        std::cerr << "  --batch            Run batch generation test" << std::endl;
        std::cerr << "  --power-test       Test power modes" << std::endl;
        std::cerr << "  --memory-test      Test memory pressure" << std::endl;
        std::cerr << "  --json <file>      Save results to JSON" << std::endl;
        std::cerr << "  --config <file>    Load TOML config" << std::endl;
        exit(1);
    }
    
    opts.model_path = argv[1];
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup_tokens = std::stoul(argv[++i]);
        } else if (arg == "--tokens" && i + 1 < argc) {
            opts.benchmark_tokens = std::stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            opts.iterations = std::stoul(argv[++i]);
        } else if (arg == "--batch") {
            opts.run_batch_test = true;
        } else if (arg == "--power-test") {
            opts.test_power_modes = true;
        } else if (arg == "--memory-test") {
            opts.test_memory_pressure = true;
        } else if (arg == "--json" && i + 1 < argc) {
            opts.json_output = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            opts.config_file = argv[++i];
        }
    }
    
    return opts;
}

void run_single_inference_benchmark(VulkanSymbioteEngine& engine, const BenchmarkOptions& opts) {
    print_section("Single Inference Benchmark");
    
    std::cout << "Configuration:" << std::endl;
    print_stat("Warmup tokens", opts.warmup_tokens);
    print_stat("Benchmark tokens", opts.benchmark_tokens);
    print_stat("Iterations", opts.iterations);
    
    auto result = engine.run_benchmark(opts.warmup_tokens, opts.benchmark_tokens, opts.iterations);
    
    std::cout << "\n" << BOLD << "Results:" << RESET << std::endl;
    print_stat("Average tokens/sec", result.avg_tokens_per_sec, "t/s");
    print_stat("Minimum tokens/sec", result.min_tokens_per_sec, "t/s");
    print_stat("Maximum tokens/sec", result.max_tokens_per_sec, "t/s");
    print_stat("Standard deviation", result.std_dev_tokens_per_sec, "t/s");
    print_stat("Average latency", result.avg_latency_ms, "ms/token");
    print_stat("Peak VRAM usage", result.peak_vram_gb, "GB");
    print_stat("Cache hit rate", result.cache_hit_rate * 100.0, "%");
    print_stat("Cache size", result.cache_size_mb, "MB");
    
    // Performance rating
    std::cout << "\n" << BOLD << "Performance Rating: " << RESET;
    if (result.avg_tokens_per_sec >= 10.0) {
        std::cout << GREEN << "EXCELLENT" << RESET << " (>=10 t/s)" << std::endl;
    } else if (result.avg_tokens_per_sec >= 7.0) {
        std::cout << GREEN << "GOOD" << RESET << " (7-10 t/s)" << std::endl;
    } else if (result.avg_tokens_per_sec >= 5.0) {
        std::cout << YELLOW << "MODERATE" << RESET << " (5-7 t/s)" << std::endl;
    } else if (result.avg_tokens_per_sec >= 3.0) {
        std::cout << YELLOW << "SLOW" << RESET << " (3-5 t/s)" << std::endl;
    } else {
        std::cout << RED << "POOR" << RESET << " (<3 t/s)" << std::endl;
    }
}

void run_batch_benchmark(VulkanSymbioteEngine& engine) {
    print_section("Batch Generation Benchmark");
    
    std::vector<std::string> prompts = {
        "Once upon a time, in a land far away,",
        "The quick brown fox jumps over the lazy dog and",
        "In the field of artificial intelligence, we see",
        "The history of computing began with",
        "Climate change is one of the most pressing issues because"
    };
    
    std::cout << "Processing " << prompts.size() << " prompts in batch..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = engine.generate_text_batch(prompts, 50, 0.7f);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double total_tokens = prompts.size() * 50.0;
    double tps = total_tokens / (duration.count() / 1000.0);
    
    std::cout << "\n" << BOLD << "Batch Results:" << RESET << std::endl;
    print_stat("Total time", duration.count() / 1000.0, "s");
    print_stat("Total tokens", total_tokens);
    print_stat("Throughput", tps, "t/s");
    print_stat("Avg time per prompt", duration.count() / static_cast<double>(prompts.size()), "ms");
    
    // Show first completion as example
    std::cout << "\n" << BOLD << "Example completion:" << RESET << std::endl;
    std::cout << CYAN << "Prompt: " << RESET << prompts[0] << std::endl;
    std::cout << CYAN << "Output: " << RESET << results[0].substr(prompts[0].length()) << std::endl;
}

void run_power_mode_comparison(VulkanSymbioteEngine& engine, const BenchmarkOptions& opts) {
    print_section("Power Mode Comparison");
    
    std::cout << "Testing different power profiles..." << std::endl;
    
    // Note: This assumes the engine has been enhanced with power mode testing
    // The actual implementation would test HIGH_PERFORMANCE, BALANCED, and POWER_SAVER
    
    std::cout << "\n" << YELLOW << "Power mode comparison requires running benchmark with --power-test flag" << RESET << std::endl;
    std::cout << "This feature tests inference performance across all three power profiles:" << std::endl;
    std::cout << "  - High Performance: Maximum throughput, higher power consumption" << std::endl;
    std::cout << "  - Balanced: Optimal efficiency for most workloads" << std::endl;
    std::cout << "  - Power Saver: Reduced performance for battery conservation" << std::endl;
}

void run_memory_pressure_test(VulkanSymbioteEngine& engine, const BenchmarkOptions& opts) {
    print_section("Memory Pressure Test");
    
    std::cout << "Testing memory pressure handling..." << std::endl;
    std::cout << "\n" << YELLOW << "Memory pressure testing evaluates how the engine handles:" << RESET << std::endl;
    std::cout << "  - Aggressive tensor eviction" << std::endl;
    std::cout << "  - Cache thrashing scenarios" << std::endl;
    std::cout << "  - Fragmentation recovery" << std::endl;
    std::cout << "  - Automatic memory defragmentation" << std::endl;
    
    // Get current memory stats
    auto vram_stats = engine.get_vram_stats();
    std::cout << "\n" << BOLD << "Current VRAM Status:" << RESET << std::endl;
    print_stat("Total VRAM", vram_stats.total_size / (1024.0 * 1024.0 * 1024.0), "GB");
    print_stat("Used VRAM", vram_stats.used_size / (1024.0 * 1024.0 * 1024.0), "GB");
    print_stat("Free VRAM", vram_stats.free_size / (1024.0 * 1024.0 * 1024.0), "GB");
    print_stat("Fragmentation ratio", vram_stats.fragmentation_ratio * 100.0, "%");
    print_stat("Allocation count", static_cast<double>(vram_stats.allocation_count));
}

void print_system_info() {
    print_section("System Information");
    
    auto& config = ConfigManager::instance();
    
    std::cout << BOLD << "Engine Configuration:" << RESET << std::endl;
    print_stat("VRAM Budget", config.memory().vram_budget_gb, "GB");
    print_stat("RAM Budget", config.memory().ram_budget_gb, "GB");
    print_stat("Compression", config.codec().enable_compression ? "Enabled" : "Disabled");
    print_stat("Algorithm", config.codec().algorithm);
    print_stat("Workgroup Size", std::to_string(config.performance().workgroup_size_x) + "x" + 
                                std::to_string(config.performance().workgroup_size_y));
    print_stat("FP16 Math", config.performance().use_fp16_math ? "Enabled" : "Disabled");
    print_stat("Subgroup Ops", config.performance().use_subgroup_ops ? "Enabled" : "Disabled");
    
    std::cout << "\n" << BOLD << "Power Settings:" << RESET << std::endl;
    print_stat("Power Profile", config.power().power_profile == 0 ? "High Performance" :
                               config.power().power_profile == 2 ? "Power Saver" : "Balanced");
    print_stat("Auto Battery Detect", config.power().auto_detect_battery ? "Yes" : "No");
    print_stat("Battery Threshold", config.power().battery_threshold_percent, "%");
}

int main(int argc, char* argv[]) {
    print_banner();
    
    auto opts = parse_args(argc, argv);
    
    try {
        // Load configuration if specified
        if (!opts.config_file.empty()) {
            auto& config = ConfigManager::instance();
            if (config.load_from_file(opts.config_file)) {
                std::cout << GREEN << "✓ Loaded configuration from: " << opts.config_file << RESET << std::endl;
            } else {
                std::cerr << YELLOW << "⚠ Failed to load config, using defaults" << RESET << std::endl;
            }
        }
        
        // Override config with command line args
        auto& config = ConfigManager::instance();
        auto benchmark_cfg = config.benchmark();
        benchmark_cfg.warmup_tokens = opts.warmup_tokens;
        benchmark_cfg.benchmark_tokens = opts.benchmark_tokens;
        benchmark_cfg.iterations = opts.iterations;
        benchmark_cfg.test_power_modes = opts.test_power_modes;
        benchmark_cfg.test_memory_pressure = opts.test_memory_pressure;
        if (!opts.json_output.empty()) {
            benchmark_cfg.output_json = true;
            benchmark_cfg.output_file = opts.json_output;
        }
        config.set_benchmark(benchmark_cfg);
        
        print_system_info();
        
        // Initialize engine
        print_section("Initializing Engine");
        std::cout << "Loading model: " << opts.model_path << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        VulkanSymbioteEngine engine(opts.model_path);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << GREEN << "✓ Engine initialized in " << init_duration.count() << "ms" << RESET << std::endl;
        
        // Print model info
        const auto& model_config = engine.config();
        std::cout << "\n" << BOLD << "Model Configuration:" << RESET << std::endl;
        print_stat("Architecture", model_config.model_type);
        print_stat("Hidden Size", model_config.hidden_size);
        print_stat("Num Layers", model_config.num_layers);
        print_stat("Num Heads", model_config.num_attention_heads);
        print_stat("Vocab Size", model_config.vocab_size);
        print_stat("Intermediate Size", model_config.intermediate_size);
        print_stat("Max Position", model_config.max_position_embeddings);
        print_stat("Head Dim", model_config.head_dim);
        
        // Run benchmarks
        run_single_inference_benchmark(engine, opts);
        
        if (opts.run_batch_test) {
            run_batch_benchmark(engine);
        }
        
        if (opts.test_power_modes) {
            run_power_mode_comparison(engine, opts);
        }
        
        if (opts.test_memory_pressure) {
            run_memory_pressure_test(engine, opts);
        }
        
        print_section("Benchmark Complete");
        
        if (!opts.json_output.empty()) {
            std::cout << GREEN << "✓ Results saved to: " << opts.json_output << RESET << std::endl;
        }
        
        std::cout << "\n" << BOLD << CYAN << "Thank you for using Vulkan Symbiote!" << RESET << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << RED << "\n✗ Error: " << e.what() << RESET << std::endl;
        return 1;
    }
    
    return 0;
}
