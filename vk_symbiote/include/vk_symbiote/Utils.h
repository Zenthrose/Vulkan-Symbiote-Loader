#pragma once
#include "Common.h"
#include <random>
#include <deque>

namespace vk_symbiote {

uint64_t hash_string(const std::string& str);
uint64_t hash_tokens(const std::vector<uint32_t>& tokens);
std::string format_bytes(uint64_t bytes);
std::string format_duration(uint64_t nanoseconds);

class RandomGenerator {
public:
    RandomGenerator(uint64_t seed = 0) : rng_(seed ? seed : std::random_device{}()) {}
    float random_float(float min = 0.0f, float max = 1.0f);
    uint32_t random_uint32(uint32_t min = 0, uint32_t max = UINT32_MAX);
    uint64_t random_uint64(uint64_t min = 0, uint64_t max = UINT64_MAX);
    void normal_distribution(float* output, size_t count, float mean = 0.0f, float stddev = 1.0f);
    void Xavier_init(float* output, size_t fan_in, size_t fan_out);
    void He_init(float* output, size_t fan_in);
private:
    std::mt19937_64 rng_;
};

void random_normal(float* data, size_t count, float mean, float stddev);
void Xavier_init(float* data, size_t fan_in, size_t fan_out);
void He_init(float* data, size_t fan_in);
RandomGenerator& get_thread_local_random();
void roPE_embedding(float* output, uint32_t position, uint32_t head_dim, float theta);
std::vector<float> create_causal_mask(uint32_t seq_len);
float softmax_inplace(float* data, size_t size);
void top_k_sampling(const float* logits, uint32_t vocab_size, uint32_t k, float temperature,
                    std::vector<uint32_t>& output_tokens, std::vector<float>& output_probs,
                    float* workspace);
uint32_t greedy_sampling(const float* logits, uint32_t vocab_size);

} // namespace vk_symbiote
