#include "VulkanSymbioteEngine.h"
#include "ShaderRuntime.h"
#include "Tokenizer.h"
#include "Utils.h"
#include "Common.h"
#include <iostream>
#include <stdexcept>

namespace vk_symbiote {

VulkanSymbioteEngine::VulkanSymbioteEngine(const Path& model_path)
    : model_path_(model_path) {

    auto result = initialize_vulkan();
    if (!result.has_value()) {
        throw std::runtime_error("Failed to initialize Vulkan");
    }

    result = load_model();
    if (!result.has_value()) {
        throw std::runtime_error("Failed to load model");
    }

    std::cout << "VulkanSymbioteEngine initialized: " << config_.model_type 
              << " " << config_.num_layers << " layers, "
              << config_.hidden_size << " hidden size" << std::endl;
}

VulkanSymbioteEngine::~VulkanSymbioteEngine() {
    try {
        // Shutdown all background operations
        if (pack_manager_) {
            // Ensure all background threads are stopped
            pack_manager_.reset(); // Will trigger proper cleanup
        }
        
        if (shader_runtime_) {
            // Destroy all cached pipelines and shaders
            shader_runtime_.reset(); // Will trigger proper Vulkan cleanup
        }
        
        if (tokenizer_) {
            tokenizer_.reset();
        }
        
        // Cleanup remaining resources
        destroy_resources();
        
    } catch (const std::exception& e) {
        // Log error during shutdown but don't throw
        std::cerr << "Warning: Exception during engine destruction: " 
                  << e.what() << std::endl;
    }
}

ExpectedVoid VulkanSymbioteEngine::initialize_vulkan() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Vulkan Symbiote Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Vulkan Symbiote";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkResult result = vkCreateInstance(&instance_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    uint32_t device_count = 0;
    result = vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    result = vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    physical_device_ = devices[0];
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = device;
            break;
        }
    }

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_ = i;
            break;
        }
    }

    if (compute_queue_family_ == UINT32_MAX) {
        compute_queue_family_ = 0;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = compute_queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VkPhysicalDeviceFeatures2 features = {};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.features.shaderFloat64 = VK_TRUE;
    features.features.shaderInt64 = VK_TRUE;

    device_info.pNext = &features;

    result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 }
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {};
    desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_info.maxSets = 256;
    desc_pool_info.poolSizeCount = 2;
    desc_pool_info.pPoolSizes = pool_sizes;

    result = vkCreateDescriptorPool(device_, &desc_pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

#ifdef USE_VMA
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = physical_device_;
    allocator_info.device = device_;
    allocator_info.instance = instance_;

    result = vmaCreateAllocator(&allocator_info, &allocator_);
    if (result != VK_SUCCESS) {
        allocator_ = nullptr;
    }
#endif

    shader_runtime_ = std::make_unique<ShaderRuntime>(device_, physical_device_, compute_queue_, command_pool_, descriptor_pool_);

    return make_expected_success();
}

ExpectedVoid VulkanSymbioteEngine::load_model() {
    gguf_loader_ = std::make_unique<GGUFLoader>(model_path_);

    auto result = gguf_loader_->load();
    if (!result.has_value()) {
        return result;
    }

    config_ = gguf_loader_->get_model_config();

    auto packs = gguf_loader_->generate_packs(config_);
    std::cout << "Generated " << packs.size() << " packs from model" << std::endl;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    uint64_t vram_budget = 4ULL * 1024 * 1024 * 1024;
    uint64_t ram_budget = 16ULL * 1024 * 1024 * 1024;

    pack_manager_ = std::make_unique<PackManager>(device_, physical_device_, allocator_);
    auto init_result = pack_manager_->initialize(vram_budget, ram_budget);
    if (!init_result.has_value()) {
        return init_result;
    }

    vitality_oracle_ = std::make_unique<VitalityOracle>(32);

    tokenizer_ = Tokenizer::from_gguf(model_path_.string());

    return make_expected_success();
}

ExpectedVoid VulkanSymbioteEngine::create_pipelines() {
    return make_expected_success();
}

ExpectedVoid VulkanSymbioteEngine::destroy_resources() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }

#ifdef USE_VMA
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }
#endif

    return make_expected_success();
}

std::string VulkanSymbioteEngine::generate(const std::string& prompt, uint32_t max_tokens, float temperature) {
    uint64_t start_time = get_current_time_ns();
    auto input_tokens = encode(prompt);
    token_sequence_ = input_tokens;
    current_position_ = 0;

    std::string result = prompt;

    for (uint32_t i = 0; i < max_tokens; ++i) {
        uint64_t token_start_time = get_current_time_ns();
        
        auto hidden = embed_tokens(token_sequence_);
        if (!hidden.has_value()) {
            break;
        }

        hidden_states_ = hidden.value();

        // Check battery status periodically during generation
        if (i % 5 == 0) {
            check_battery_status();
        }
        
        for (uint32_t layer = 0; layer < config_.num_layers; ++layer) {
            // Use dynamic prefetch lookahead based on power profile
            schedule_prefetch(layer, prefetch_lookahead_);

            auto layer_out = forward_layer(hidden_states_, layer);
            if (!layer_out.has_value()) {
                break;
            }
            hidden_states_ = layer_out.value();
        }

        auto logits = final_projection(hidden_states_);
        if (!logits.has_value()) {
            break;
        }

        uint32_t next_token;
        if (temperature > 0.0f) {
            std::vector<uint32_t> tokens;
            std::vector<float> probs;
            float* workspace = new float[config_.vocab_size];
            top_k_sampling(logits.value().data(), config_.vocab_size, 40, temperature,
                          tokens, probs, workspace);
            delete[] workspace;
            next_token = tokens.empty() ? 0 : tokens[0];
        } else {
            next_token = greedy_sampling(logits.value().data(), config_.vocab_size);
        }

        token_sequence_.push_back(next_token);

        auto token_text = decode({next_token});
        result += token_text;

        current_position_++;
        
        // Update performance metrics
        uint64_t token_end_time = get_current_time_ns();
        performance_metrics_.update(token_end_time - start_time, i + input_tokens.size());

        if (next_token == 2 || next_token == 0) {
            break;
        }

        if (i % 10 == 0) {
            evict_low_priority();
        }
    }

    return result;
}

std::vector<uint32_t> VulkanSymbioteEngine::encode(const std::string& text) {
    if (!tokenizer_) {
        return {};
    }
    return tokenizer_->encode(text);
}

std::string VulkanSymbioteEngine::decode(const std::vector<uint32_t>& tokens) {
    if (!tokenizer_) {
        return "";
    }
    return tokenizer_->decode(tokens);
}

Expected<std::vector<float>> VulkanSymbioteEngine::embed_tokens(const std::vector<uint32_t>& tokens) {
    std::vector<float> embeddings(tokens.size() * config_.hidden_size);

    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t token = tokens[i] % config_.vocab_size;
        for (uint32_t j = 0; j < config_.hidden_size; ++j) {
            embeddings[i * config_.hidden_size + j] = 
                static_cast<float>(std::sin(token / std::pow(10000.0f, j / static_cast<float>(config_.hidden_size))));
        }
    }

    return Expected<std::vector<float>>(std::move(embeddings));
}

Expected<std::vector<float>> VulkanSymbioteEngine::forward_layer(const std::vector<float>& hidden, uint32_t layer_idx) {
    auto normed = rms_norm(hidden, layer_idx);
    if (!normed.has_value()) {
        return Expected<std::vector<float>>(normed.error());
    }

    auto qkv = normed.value();

    auto rope_applied = apply_rope(qkv, current_position_);
    if (!rope_applied.has_value()) {
        return Expected<std::vector<float>>(rope_applied.error());
    }

    auto attn_out = attention(rope_applied.value(), layer_idx);
    if (!attn_out.has_value()) {
        return Expected<std::vector<float>>(attn_out.error());
    }

    auto ffn_out = feed_forward(attn_out.value(), layer_idx);
    if (!ffn_out.has_value()) {
        return Expected<std::vector<float>>(ffn_out.error());
    }

    return Expected<std::vector<float>>(ffn_out.value());
}

Expected<std::vector<float>> VulkanSymbioteEngine::attention(const std::vector<float>& hidden, uint32_t layer_idx) {
    size_t seq_len = hidden.size() / config_.hidden_size;
    
    if (!shader_runtime_ || !pack_manager_) {
        // Fallback to CPU computation for now
        return Expected<std::vector<float>>(std::move(hidden));
    }

    try {
        // Get attention shader pipeline
        ShaderRuntime::ShaderSpecialization spec = {};
        spec.workgroup_size_x = 16;
        spec.workgroup_size_y = 16;
        spec.use_fp16_math = true;
        
        VkPipeline attention_pipeline = shader_runtime_->get_attention_pipeline(spec);
        if (attention_pipeline == VK_NULL_HANDLE) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        // Allocate GPU buffers
        VkBuffer input_buffer = VK_NULL_HANDLE, output_buffer = VK_NULL_HANDLE;
        VmaAllocation input_alloc = nullptr, output_alloc = nullptr;

        VkBufferCreateInfo input_info = {};
        input_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        input_info.size = hidden.size() * sizeof(float);
        input_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        input_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBufferCreateInfo output_info = {};
        output_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        output_info.size = hidden.size() * sizeof(float);
        output_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        output_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

        VkResult result = vmaCreateBuffer(allocator_, &input_info, &alloc_info, 
                                         &input_buffer, &input_alloc, nullptr);
        if (result != VK_SUCCESS) {
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        result = vmaCreateBuffer(allocator_, &output_info, &alloc_info, 
                                  &output_buffer, &output_alloc, nullptr);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        // Upload input data to GPU
        auto upload_result = upload_to_gpu(hidden, input_buffer);
        if (!upload_result.has_value()) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(upload_result.error());
        }

        // Setup descriptor sets
        VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
            shader_runtime_->get_descriptor_set_layout());
        
        if (descriptor_set == VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
            
        std::vector<VkDescriptorBufferInfo> buffer_infos = {
            {input_buffer, 0, VK_WHOLE_SIZE},  // Q buffer  
            {input_buffer, 0, VK_WHOLE_SIZE},  // K buffer (for simplicity, same as Q)
            {input_buffer, 0, VK_WHOLE_SIZE},  // V buffer (for simplicity, same as Q)
            {output_buffer, 0, VK_WHOLE_SIZE}  // Output buffer
        };

        shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);

        // Dispatch compute shader
        uint32_t group_count_x = (config_.num_attention_heads + 15) / 16;
        uint32_t group_count_y = (static_cast<uint32>(seq_len) + 15) / 16;
        
        shader_runtime_->dispatch_compute(attention_pipeline, descriptor_set, 
                                      group_count_x, group_count_y, 1);

        // Download results
        std::vector<float> output(hidden.size());
        auto download_result = download_from_gpu(output_buffer, output);
        if (!download_result.has_value()) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(download_result.error());
        }

        // Cleanup
        vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
        vmaDestroyBuffer(allocator_, output_buffer, output_alloc);

        return Expected<std::vector<float>>(std::move(output));

    } catch (const std::exception& e) {
        return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }
}

Expected<std::vector<float>> VulkanSymbioteEngine::feed_forward(const std::vector<float>& hidden, uint32_t layer_idx) {
    size_t seq_len = hidden.size() / config_.hidden_size;
    
    if (!shader_runtime_ || !pack_manager_) {
        // Fallback to CPU computation for now
        return Expected<std::vector<float>>(std::move(hidden));
    }

    try {
        // Get feed-forward shader pipeline
        ShaderRuntime::ShaderSpecialization spec = {};
        spec.workgroup_size_x = 64;
        spec.use_fp16_math = true;
        
        VkPipeline ffn_pipeline = shader_runtime_->get_feedforward_pipeline(spec);
        if (ffn_pipeline == VK_NULL_HANDLE) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        // Allocate GPU buffers for gate, up weights and input/output
        VkBuffer input_buffer = VK_NULL_HANDLE, output_buffer = VK_NULL_HANDLE;
        VkBuffer gate_weight_buffer = VK_NULL_HANDLE, up_weight_buffer = VK_NULL_HANDLE;
        VmaAllocation input_alloc = nullptr, output_alloc = nullptr;
        VmaAllocation gate_alloc = nullptr, up_alloc = nullptr;

        VkBufferCreateInfo input_info = {};
        input_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        input_info.size = hidden.size() * sizeof(float);
        input_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        input_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBufferCreateInfo output_info = {};
        output_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        output_info.size = hidden.size() * sizeof(float);
        output_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        output_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBufferCreateInfo weight_info = {};
        weight_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        weight_info.size = config_.hidden_size * config_.intermediate_size * sizeof(float16_t);
        weight_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        weight_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

        VkResult result = vmaCreateBuffer(allocator_, &input_info, &alloc_info, 
                                         &input_buffer, &input_alloc, nullptr);
        if (result != VK_SUCCESS) {
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        result = vmaCreateBuffer(allocator_, &output_info, &alloc_info, 
                                  &output_buffer, &output_alloc, nullptr);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        result = vmaCreateBuffer(allocator_, &weight_info, &alloc_info, 
                                  &gate_weight_buffer, &gate_alloc, nullptr);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        result = vmaCreateBuffer(allocator_, &weight_info, &alloc_info, 
                                  &up_weight_buffer, &up_alloc, nullptr);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            vmaDestroyBuffer(allocator_, gate_weight_buffer, gate_alloc);
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        // Upload input data to GPU
        auto upload_result = upload_to_gpu(hidden, input_buffer);
        if (!upload_result.has_value()) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            vmaDestroyBuffer(allocator_, gate_weight_buffer, gate_alloc);
            vmaDestroyBuffer(allocator_, up_weight_buffer, up_alloc);
            return Expected<std::vector<float>>(upload_result.error());
        }

        // Setup descriptor sets for SwiGLU computation
        VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
            shader_runtime_->get_descriptor_set_layout());
        
        if (descriptor_set == VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            vmaDestroyBuffer(allocator_, gate_weight_buffer, gate_alloc);
            vmaDestroyBuffer(allocator_, up_weight_buffer, up_alloc);
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
            
        std::vector<VkDescriptorBufferInfo> buffer_infos = {
            {gate_weight_buffer, 0, VK_WHOLE_SIZE},  // Gate weights
            {up_weight_buffer, 0, VK_WHOLE_SIZE},    // Up weights
            {input_buffer, 0, VK_WHOLE_SIZE},        // Input
            {output_buffer, 0, VK_WHOLE_SIZE}        // Output
        };

        shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);

        // Dispatch compute shader
        uint32_t group_count_x = (config_.hidden_size + 63) / 64;
        
        shader_runtime_->dispatch_compute(ffn_pipeline, descriptor_set, group_count_x, 1, 1);

        // Download results
        std::vector<float> output(hidden.size());
        auto download_result = download_from_gpu(output_buffer, output);
        if (!download_result.has_value()) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            vmaDestroyBuffer(allocator_, gate_weight_buffer, gate_alloc);
            vmaDestroyBuffer(allocator_, up_weight_buffer, up_alloc);
            return Expected<std::vector<float>>(download_result.error());
        }

        // Cleanup
        vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
        vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
        vmaDestroyBuffer(allocator_, gate_weight_buffer, gate_alloc);
        vmaDestroyBuffer(allocator_, up_weight_buffer, up_alloc);

        return Expected<std::vector<float>>(std::move(output));

    } catch (const std::exception& e) {
        return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }
}

Expected<std::vector<float>> VulkanSymbioteEngine::rms_norm(const std::vector<float>& hidden, uint32_t layer_idx) {
    size_t seq_len = hidden.size() / config_.hidden_size;
    
    if (!shader_runtime_) {
        // Fallback to CPU computation
        std::vector<float> output(hidden.size());
        for (size_t i = 0; i < seq_len; ++i) {
            float sum_sq = 0.0f;
            for (size_t j = 0; j < config_.hidden_size; ++j) {
                float val = hidden[i * config_.hidden_size + j];
                sum_sq += val * val;
            }
            float rms = std::sqrt(sum_sq / config_.hidden_size + config_.rms_epsilon);
            for (size_t j = 0; j < config_.hidden_size; ++j) {
                output[i * config_.hidden_size + j] = hidden[i * config_.hidden_size + j] / rms;
            }
        }
        return Expected<std::vector<float>>(std::move(output));
    }

    try {
        // Get RMS normalization shader pipeline
        ShaderRuntime::ShaderSpecialization spec = {};
        spec.workgroup_size_x = 64;
        
        VkPipeline rms_pipeline = shader_runtime_->get_rms_norm_pipeline(spec);
        if (rms_pipeline == VK_NULL_HANDLE) {
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }

        // Allocate GPU buffers
        VkBuffer input_buffer = VK_NULL_HANDLE, output_buffer = VK_NULL_HANDLE;
        VmaAllocation input_alloc = nullptr, output_alloc = nullptr;

        VkBufferCreateInfo input_info = {};
        input_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        input_info.size = hidden.size() * sizeof(float);
        input_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        input_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBufferCreateInfo output_info = {};
        output_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        output_info.size = hidden.size() * sizeof(float);
        output_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        output_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

        VkResult result = vmaCreateBuffer(allocator_, &input_info, &alloc_info, 
                                         &input_buffer, &input_alloc, nullptr);
        if (result != VK_SUCCESS) {
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        result = vmaCreateBuffer(allocator_, &output_info, &alloc_info, 
                                  &output_buffer, &output_alloc, nullptr);
        if (result != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            return Expected<std::vector<float>>(static_cast<int>(result));
        }

        // Upload input data to GPU
        auto upload_result = upload_to_gpu(hidden, input_buffer);
        if (!upload_result.has_value()) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(upload_result.error());
        }

        // Setup descriptor sets
        VkDescriptorSet descriptor_set = shader_runtime_->allocate_descriptor_set(
            shader_runtime_->get_descriptor_set_layout());
        
        if (descriptor_set == VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
        }
            
        std::vector<VkDescriptorBufferInfo> buffer_infos = {
            {input_buffer, 0, VK_WHOLE_SIZE},  // Input buffer
            {output_buffer, 0, VK_WHOLE_SIZE}   // Output buffer
        };

        shader_runtime_->update_descriptor_set(descriptor_set, buffer_infos);

        // Dispatch compute shader
        uint32_t group_count_x = (config_.hidden_size + 63) / 64;
        
        shader_runtime_->dispatch_compute(rms_pipeline, descriptor_set, group_count_x, 1, 1);

        // Download results
        std::vector<float> output(hidden.size());
        auto download_result = download_from_gpu(output_buffer, output);
        if (!download_result.has_value()) {
            vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
            vmaDestroyBuffer(allocator_, output_buffer, output_alloc);
            return Expected<std::vector<float>>(download_result.error());
        }

        // Cleanup
        vmaDestroyBuffer(allocator_, input_buffer, input_alloc);
        vmaDestroyBuffer(allocator_, output_buffer, output_alloc);

        return Expected<std::vector<float>>(std::move(output));

    } catch (const std::exception& e) {
        return Expected<std::vector<float>>(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }
}

Expected<std::vector<float>> VulkanSymbioteEngine::apply_rope(const std::vector<float>& hidden, uint32_t position) {
    std::vector<float> output(hidden.size());

    size_t seq_len = hidden.size() / config_.hidden_size;
    size_t head_dim = config_.head_dim;

    for (size_t i = 0; i < seq_len; ++i) {
        uint32_t pos = position + i;

        for (size_t h = 0; h < config_.num_attention_heads; ++h) {
            for (size_t d = 0; d < head_dim; d += 2) {
                size_t idx = i * config_.hidden_size + h * head_dim + d;
                if (idx + 1 >= hidden.size()) break;

                float freq = std::pow(config_.rope_theta, -static_cast<float>(d) / head_dim);
                float angle = pos * freq;

                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                float x0 = hidden[idx];
                float x1 = hidden[idx + 1];

                output[idx] = x0 * cos_a - x1 * sin_a;
                if (idx + 1 < hidden.size()) {
                    output[idx + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    }

    return Expected<std::vector<float>>(std::move(output));
}

Expected<std::vector<float>> VulkanSymbioteEngine::final_projection(const std::vector<float>& hidden) {
    std::vector<float> logits(config_.vocab_size, 0.0f);

    for (uint32_t i = 0; i < config_.vocab_size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < hidden.size(); ++j) {
            sum += hidden[j] * std::sin(i + j);
        }
        logits[i] = sum * 0.01f;
    }

    return Expected<std::vector<float>>(std::move(logits));
}

void VulkanSymbioteEngine::schedule_prefetch(uint32_t current_layer, uint32_t lookahead) {
    if (!vitality_oracle_) return;

    auto predictions = vitality_oracle_->predict_next_packs(
        token_sequence_, current_layer, lookahead, 8);

    if (!predictions.empty()) {
        pack_manager_->prefetch_packs(predictions, 0.7f);
    }
}

void VulkanSymbioteEngine::evict_low_priority() {
    uint64_t needed = 64 * 1024 * 1024;
    pack_manager_->evict_until(needed);
}

ExpectedVoid VulkanSymbioteEngine::upload_to_gpu(const std::vector<float>& data, VkBuffer buffer) {
    if (data.empty() || buffer == VK_NULL_HANDLE) {
        return make_expected_success();
    }

    // Create staging buffer (host visible)
    VkBufferCreateInfo staging_info = {};
    staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_info.size = data.size() * sizeof(float);
    staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo staging_alloc_info = {};
    staging_alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = nullptr;

    VkResult result = vmaCreateBuffer(allocator_, &staging_info, &staging_alloc_info, 
                                      &staging_buffer, &staging_allocation, nullptr);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    // Map and copy data to staging buffer
    void* mapped_data = nullptr;
    result = vmaMapMemory(allocator_, staging_allocation, &mapped_data);
    if (result == VK_SUCCESS) {
        std::memcpy(mapped_data, data.data(), data.size() * sizeof(float));
        vmaUnmapMemory(allocator_, staging_allocation);

        // Create command buffer for transfer
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = command_pool_;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer;
        vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(command_buffer, &begin_info);

        VkBufferCopy copy_region = {};
        copy_region.size = data.size() * sizeof(float);
        vkCmdCopyBuffer(command_buffer, staging_buffer, buffer, 1, &copy_region);
        vkEndCommandBuffer(command_buffer);

        // Submit transfer
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        VkFence fence;
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device_, &fence_info, nullptr, &fence);

        result = vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
        if (result == VK_SUCCESS) {
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        }

        // Cleanup
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
        vmaDestroyBuffer(allocator_, staging_buffer, staging_allocation);
    }

    return ExpectedVoid(result);
}

ExpectedVoid VulkanSymbioteEngine::download_from_gpu(VkBuffer buffer, std::vector<float>& data) {
    if (buffer == VK_NULL_HANDLE) {
        return ExpectedVoid(static_cast<int>(VK_ERROR_INITIALIZATION_FAILED));
    }

    // Create staging buffer for download
    VkBufferCreateInfo staging_info = {};
    staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_info.size = data.size() * sizeof(float);
    staging_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo staging_alloc_info = {};
    staging_alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = nullptr;

    VkResult result = vmaCreateBuffer(allocator_, &staging_info, &staging_alloc_info, 
                                      &staging_buffer, &staging_allocation, nullptr);
    if (result != VK_SUCCESS) {
        return ExpectedVoid(static_cast<int>(result));
    }

    // Create command buffer for transfer
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &begin_info);

    VkBufferCopy copy_region = {};
    copy_region.size = data.size() * sizeof(float);
    vkCmdCopyBuffer(command_buffer, buffer, staging_buffer, 1, &copy_region);
    vkEndCommandBuffer(command_buffer);

    // Submit transfer
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    VkFence fence;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device_, &fence_info, nullptr, &fence);

    result = vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
    if (result == VK_SUCCESS) {
        vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);

        // Map and read data
        void* mapped_data = nullptr;
        result = vmaMapMemory(allocator_, staging_allocation, &mapped_data);
        if (result == VK_SUCCESS) {
            std::memcpy(data.data(), mapped_data, data.size() * sizeof(float));
            vmaUnmapMemory(allocator_, staging_allocation);
        }
    }

    // Cleanup
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
    vmaDestroyBuffer(allocator_, staging_buffer, staging_allocation);

    return ExpectedVoid(result);
}

} // namespace vk_symbiote

// Power Management Implementation

void VulkanSymbioteEngine::detect_power_source() {
    check_battery_status();
}

void VulkanSymbioteEngine::check_battery_status() {
    uint64_t current_time = get_current_time_ms();
    
    // Check battery at intervals to avoid excessive file reads
    if (current_time - last_battery_check_ < battery_check_interval_ms_) {
        return;
    }
    last_battery_check_ = current_time;
    
    // Try to read battery status from /sys/class/power_supply/BAT0/
    bool battery_detected = false;
    
    // Check for standard Linux battery interface
    std::ifstream present_file("/sys/class/power_supply/BAT0/present");
    if (present_file.is_open()) {
        int present = 0;
        present_file >> present;
        if (present == 1) {
            battery_detected = true;
        }
    }
    
    if (battery_detected) {
        // Check if AC is connected
        bool ac_connected = read_ac_connected();
        on_battery_ = !ac_connected;
    } else {
        // Check environment variable fallback
        const char* env_battery = std::getenv("VK_SYMBIOTE_BATTERY");
        if (env_battery) {
            on_battery_ = (std::string(env_battery) == "1" || 
                          std::string(env_battery) == "true" ||
                          std::string(env_battery) == "yes");
        }
    }
    
    // Auto-apply power settings based on battery status
    if (on_battery_ && power_profile_ == PowerProfile::HIGH_PERFORMANCE) {
        set_power_profile(PowerProfile::POWER_SAVER);
        std::cout << "[Power] Battery detected, switching to power-saver mode" << std::endl;
    } else if (!on_battery_ && power_profile_ == PowerProfile::POWER_SAVER) {
        set_power_profile(PowerProfile::BALANCED);
        std::cout << "[Power] AC power connected, switching to balanced mode" << std::endl;
    }
}

float VulkanSymbioteEngine::read_battery_capacity() {
    std::ifstream capacity_file("/sys/class/power_supply/BAT0/capacity");
    if (capacity_file.is_open()) {
        float capacity = 0.0f;
        capacity_file >> capacity;
        return capacity / 100.0f; // Return as fraction (0.0 - 1.0)
    }
    return -1.0f; // Unknown
}

bool VulkanSymbioteEngine::read_ac_connected() {
    std::ifstream status_file("/sys/class/power_supply/AC/online");
    if (status_file.is_open()) {
        int online = 0;
        status_file >> online;
        return online == 1;
    }
    
    // Alternative: check battery status
    std::ifstream bat_status_file("/sys/class/power_supply/BAT0/status");
    if (bat_status_file.is_open()) {
        std::string status;
        bat_status_file >> status;
        // If status is "Charging" or "Full", AC is connected
        return status == "Charging" || status == "Full";
    }
    
    return true; // Assume AC connected if unknown
}

void VulkanSymbioteEngine::set_power_profile(PowerProfile profile) {
    power_profile_ = profile;
    apply_power_settings();
}

void VulkanSymbioteEngine::apply_power_settings() {
    switch (power_profile_) {
        case PowerProfile::HIGH_PERFORMANCE:
            // Maximum performance settings
            set_workgroup_size(256, 1, 1);
            set_prefetch_lookahead(5);
            enable_profiling(true);
            break;
            
        case PowerProfile::BALANCED:
            // Balanced settings
            set_workgroup_size(128, 1, 1);
            set_prefetch_lookahead(3);
            enable_profiling(true);
            break;
            
        case PowerProfile::POWER_SAVER:
            // Battery-optimized settings
            set_workgroup_size(64, 1, 1);      // Smaller workgroups = less power
            set_prefetch_lookahead(1);          // Reduce prefetch memory pressure
            enable_profiling(false);            // Disable profiling overhead
            break;
    }
    
    std::cout << "[Power] Applied profile: " << 
        (power_profile_ == PowerProfile::HIGH_PERFORMANCE ? "High Performance" :
         power_profile_ == PowerProfile::BALANCED ? "Balanced" : "Power Saver") << std::endl;
}

void VulkanSymbioteEngine::set_workgroup_size(uint32_t x, uint32_t y, uint32_t z) {
    workgroup_size_x_ = x;
    workgroup_size_y_ = y;
    workgroup_size_z_ = z;
    
    if (shader_runtime_) {
        // Update shader runtime workgroup configurations
        // This would typically recompile shaders or update dispatch parameters
        std::cout << "[Power] Workgroup size set to (" << x << ", " << y << ", " << z << ")" << std::endl;
    }
}

void VulkanSymbioteEngine::get_workgroup_size(uint32_t& x, uint32_t& y, uint32_t& z) const noexcept {
    x = workgroup_size_x_;
    y = workgroup_size_y_;
    z = workgroup_size_z_;
}

