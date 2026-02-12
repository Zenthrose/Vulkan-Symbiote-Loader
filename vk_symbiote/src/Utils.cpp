#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace vk_symbiote {

uint64_t hash_string(const std::string& str) {
    uint64_t hash = 1469598103934665603ULL;
    for (char c : str) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t hash_tokens(const std::vector<uint32_t>& tokens) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint32_t token : tokens) {
        hash ^= token;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string format_bytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

std::string format_duration(uint64_t nanoseconds) {
    if (nanoseconds < 1000) {
        return std::to_string(nanoseconds) + "ns";
    } else if (nanoseconds < 1000000) {
        return std::to_string(nanoseconds / 1000) + "us";
    } else if (nanoseconds < 1000000000) {
        return std::to_string(nanoseconds / 1000000) + "ms";
    } else {
        return std::to_string(nanoseconds / 1000000000) + "s";
    }
}

RandomGenerator& get_thread_local_random() {
    thread_local RandomGenerator rng;
    return rng;
}

float RandomGenerator::random_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng_);
}

uint32_t RandomGenerator::random_uint32(uint32_t min, uint32_t max) {
    std::uniform_int_distribution<uint32_t> dist(min, max);
    return dist(rng_);
}

uint64_t RandomGenerator::random_uint64(uint64_t min, uint64_t max) {
    std::uniform_int_distribution<uint64_t> dist(min, max);
    return dist(rng_);
}

void RandomGenerator::normal_distribution(float* output, size_t count, float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < count; ++i) {
        output[i] = dist(rng_);
    }
}

void RandomGenerator::Xavier_init(float* output, size_t fan_in, size_t fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    normal_distribution(output, fan_in * fan_out, 0.0f, limit);
}

void RandomGenerator::He_init(float* output, size_t fan_in) {
    float stddev = std::sqrt(2.0f / fan_in);
    normal_distribution(output, fan_in, 0.0f, stddev);
}

void random_normal(float* data, size_t count, float mean, float stddev) {
    get_thread_local_random().normal_distribution(data, count, mean, stddev);
}

void Xavier_init(float* data, size_t fan_in, size_t fan_out) {
    get_thread_local_random().Xavier_init(data, fan_in, fan_out);
}

void He_init(float* data, size_t fan_in) {
    get_thread_local_random().He_init(data, fan_in);
}

void roPE_embedding(float* output, uint32_t position, uint32_t head_dim, float theta) {
    for (uint32_t i = 0; i < head_dim; i += 2) {
        float freq = std::pow(theta, -static_cast<float>(i) / head_dim);
        float angle = position * freq;
        output[i] = std::cos(angle);
        if (i + 1 < head_dim) {
            output[i + 1] = std::sin(angle);
        }
    }
}

std::vector<float> create_causal_mask(uint32_t seq_len) {
    std::vector<float> mask(seq_len * seq_len, -1e9f);
    for (uint32_t i = 0; i < seq_len; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            mask[i * seq_len + j] = 0.0f;
        }
    }
    return mask;
}

float softmax_inplace(float* data, size_t size) {
    float max_val = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    if (sum > 0) {
        for (size_t i = 0; i < size; ++i) {
            data[i] /= sum;
        }
    }
    return sum;
}

void top_k_sampling(const float* logits, uint32_t vocab_size, uint32_t k, float temperature,
                    std::vector<uint32_t>& output_tokens, std::vector<float>& output_probs,
                    float* workspace) {
    std::vector<uint32_t> indices(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        indices[i] = i;
    }
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](uint32_t a, uint32_t b) {
                          return logits[a] > logits[b];
                      });
    float max_logit = logits[indices[0]];
    float sum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        workspace[i] = std::exp((logits[indices[i]] - max_logit) / temperature);
        sum += workspace[i];
    }
    for (uint32_t i = 0; i < k; ++i) {
        workspace[i] /= sum;
    }
    float r = get_thread_local_random().random_float();
    float cumsum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        cumsum += workspace[i];
        if (r < cumsum) {
            output_tokens.push_back(indices[i]);
            output_probs.push_back(workspace[i]);
            return;
        }
    }
    output_tokens.push_back(indices[0]);
    output_probs.push_back(workspace[0]);
}

uint32_t greedy_sampling(const float* logits, uint32_t vocab_size) {
    uint32_t max_idx = 0;
    float max_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace vk_symbiote
