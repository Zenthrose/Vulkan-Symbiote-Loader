#include "../include/vk_symbiote/VulkanSymbioteEngine.h"
#include <iostream>

using namespace vk_symbiote;

int main() {
    try {
        std::filesystem::path model_path = "test_model.gguf";
        
        VulkanSymbioteEngine engine(model_path);
        
        std::string prompt = "Test prompt";
        std::string result = engine.generate(prompt, 10, 0.7f);
        
        std::cout << "Result: " << result << std::endl;
        
        uint32_t iterations = 100;
        for (uint32_t i = 0; i < iterations; ++i) {
            engine.generate("Benchmark test", 5, 0.8f);
        }
        
        return 0;
    } catch (...) {
        return -1;
    }
}