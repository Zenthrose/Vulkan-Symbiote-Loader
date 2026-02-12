#include "../include/vk_symbiote/VulkanSymbioteEngine.h"
#include "../include/vk_symbiote/ConfigManager.h"
#include <iostream>

using namespace vk_symbiote;

int main(int argc, char* argv[]) {
    try {
        // Load configuration from command line
        ConfigManager::instance().load_from_args(argc, argv);
        
        // Get model path from config or command line
        std::string model_path = ConfigManager::instance().model_path();
        if (model_path.empty()) {
            std::cerr << "Error: Model path not specified. Use --model <path>" << std::endl;
            return 1;
        }
        
        std::cout << "Initializing Vulkan Symbiote Engine..." << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        
        // Display configuration
        ConfigManager::instance().print_config();
        
        VulkanSymbioteEngine engine(model_path);
        
        std::string prompt = "Hello, Vulkan Symbiote!";
        uint32_t max_tokens = 256;
        float temperature = 0.7f;
        
        std::cout << "Generating text..." << std::endl;
        std::string result = engine.generate(prompt, max_tokens, temperature);
        
        std::cout << std::endl << "Generated " << result.length() << " characters" << std::endl;
        std::cout << "Result: " << result << std::endl;
        
        // Display performance metrics
        const auto& metrics = engine.get_performance_metrics();
        if (metrics.total_inferences > 0) {
            float avg_time_ms = static_cast<float>(metrics.total_inference_time_ns) / 1e6f / metrics.total_inferences;
            std::cout << "Performance: " << metrics.average_tokens_per_second << " tokens/sec" << std::endl;
            std::cout << "Average time: " << avg_time_ms << " ms/token" << std::endl;
            std::cout << "GPU utilization: " << metrics.gpu_utilization_percent << "%" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}