#include "VitalityOracle.h"
#include <algorithm>
#include <cmath>

namespace vk_symbiote {

VitalityOracle::VitalityOracle(uint32 max_packs_in_memory) : rng_(std::random_device{}()) {
    model_.input_size = 64; model_.hidden_size = 16;
    model_.w1.resize(model_.hidden_size * model_.input_size);
    model_.b1.resize(model_.hidden_size);
    model_.w2.resize(model_.hidden_size);
    model_.b2.resize(1);

    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    for (auto& w : model_.w1) w = dist(rng_);
    for (auto& b : model_.b1) b = dist(rng_);
    for (auto& w : model_.w2) w = dist(rng_);
    model_.b2[0] = 0.0f;
}

VitalityOracle::~VitalityOracle() { recent_accesses_.clear(); }

VitalityScore VitalityOracle::score_pack(const PackMetadata& pack, const std::vector<uint32_t>& tokens,
                                        uint32 current_layer, const HardwareTelemetry& telemetry) {
    VitalityScore score;
    score.relevance = compute_relevance(pack, tokens, current_layer);
    score.hardware_score = compute_hardware_score(pack, telemetry);
    score.temporal_score = compute_temporal_score(pack.pack_id);
    score.priority_bonus = compute_priority_bonus(pack);
    auto features = extract_features(pack, tokens, current_layer, telemetry);
    score.confidence = mlp_forward(features);
    score.confidence = std::clamp(score.confidence, 0.0f, 1.0f);
    return score;
}

std::vector<uint64> VitalityOracle::predict_next_packs(const std::vector<uint32_t>& tokens,
                                                       uint32 current_layer, uint32 lookahead, uint32 count) {
    std::vector<uint64> predictions;
    std::vector<std::pair<uint64, float>> scored;

    for (uint32 offset = 0; offset < lookahead; ++offset) {
        uint32 layer = current_layer + offset;
        if (layer >= 100) break;
        float base_score = 1.0f - static_cast<float>(offset) / lookahead * 0.3f;
        for (uint32 head = 0; head < 32 && scored.size() < count * 2; ++head) {
            scored.push_back({layer * 1000 + head * 10, base_score});
            scored.push_back({layer * 1000 + head * 10 + 1, base_score * 0.9f});
            scored.push_back({layer * 1000 + head * 10 + 2, base_score * 0.9f});
        }
        scored.push_back({(current_layer + offset) * 1000 + 500, 0.7f - static_cast<float>(offset) / lookahead * 0.2f});
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (uint32 i = 0; i < std::min(count, static_cast<uint32>(scored.size())); ++i) {
        predictions.push_back(scored[i].first);
    }
    return predictions;
}

void VitalityOracle::update_telemetry(const HardwareTelemetry& telemetry) { (void)telemetry; }

void VitalityOracle::record_access(uint64 pack_id, bool was_used, float predicted_score) {
    AccessRecord record{pack_id, get_current_time_ns(), was_used, predicted_score};
    recent_accesses_.push_back(record);
    if (recent_accesses_.size() > max_access_history_) recent_accesses_.pop_front();
    total_predictions_++;
    if (was_used) correct_predictions_++;
    if (total_predictions_ > 0) hit_rate_ = static_cast<float>(correct_predictions_) / static_cast<float>(total_predictions_);
}

void VitalityOracle::update_model() {
    if (recent_accesses_.size() < 32) return;
    float learning_rate = 0.001f;
    for (const auto& record : recent_accesses_) {
        float error = record.was_used ? (1.0f - record.predicted_score) : (-record.predicted_score);
        for (uint32 i = 0; i < model_.hidden_size; ++i) {
            model_.b1[i] += error * learning_rate;
        }
    }
    recent_accesses_.clear();
}

std::vector<float> VitalityOracle::extract_features(const PackMetadata& pack,
                                                   const std::vector<uint32_t>& tokens,
                                                   uint32 current_layer, const HardwareTelemetry& telemetry) {
    std::vector<float> features(model_.input_size, 0.0f);
    features[0] = static_cast<float>(pack.layer_idx) / 100.0f;
    features[1] = static_cast<float>(pack.head_idx) / 32.0f;
    features[2] = static_cast<float>(pack.type) / 255.0f;
    features[3] = pack.base_priority;
    features[4] = compute_relevance(pack, tokens, current_layer);
    features[5] = compute_hardware_score(pack, telemetry);
    features[6] = compute_temporal_score(pack.pack_id);
    features[7] = static_cast<float>(tokens.size()) / 2048.0f;
    for (uint32 i = 9; i < model_.input_size; ++i) {
        features[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.5f + 0.5f;
    }
    return features;
}

float VitalityOracle::mlp_forward(const std::vector<float>& input) {
    std::vector<float> hidden(model_.hidden_size, 0.0f);
    for (uint32 i = 0; i < model_.hidden_size; ++i) {
        float sum = model_.b1[i];
        for (uint32 j = 0; j < model_.input_size; ++j) {
            sum += input[j] * model_.w1[i * model_.input_size + j];
        }
        hidden[i] = std::tanh(sum);
    }
    float output = model_.b2[0];
    for (uint32 i = 0; i < model_.hidden_size; ++i) {
        output += hidden[i] * model_.w2[i];
    }
    return 1.0f / (1.0f + std::exp(-output));
}

void VitalityOracle::mlp_backward(const std::vector<float>& input, const std::vector<float>& gradient, float learning_rate) {
    (void)input; (void)gradient; (void)learning_rate;
}

uint64_t VitalityOracle::hash_tokens(const std::vector<uint32_t>& tokens) const {
    uint64_t hash = 1469598103934665603ULL;
    for (uint32_t token : tokens) {
        hash ^= token;
        hash *= 1099511628211ULL;
    }
    return hash;
}

float VitalityOracle::compute_relevance(const PackMetadata& pack, const std::vector<uint32_t>& tokens, uint32 current_layer) {
    float layer_proximity = 1.0f - std::abs(static_cast<float>(pack.layer_idx) - static_cast<float>(current_layer)) / 32.0f;
    layer_proximity = std::max(0.0f, layer_proximity);
    float type_weight = (pack.type == PackType::ATTENTION_Q || pack.type == PackType::ATTENTION_K || pack.type == PackType::ATTENTION_V) ? 1.0f : 0.8f;
    float position_weight = 1.0f - static_cast<float>(tokens.size()) / 2048.0f * 0.3f;
    return layer_proximity * type_weight * position_weight;
}

float VitalityOracle::compute_hardware_score(const PackMetadata& pack, const HardwareTelemetry& telemetry) {
    (void)pack;
    float score = 1.0f;
    if (telemetry.gpu_utilization > 0.9f) score *= 0.5f;
    else if (telemetry.gpu_utilization < 0.5f) score *= 1.2f;
    if (telemetry.memory_bandwidth > 0.8f) score *= 0.7f;
    score *= telemetry.thermal_throttle;
    return std::clamp(score, 0.0f, 1.0f);
}

float VitalityOracle::compute_temporal_score(uint64 pack_id) {
    for (const auto& record : recent_accesses_) {
        if (record.pack_id == pack_id) {
            float age = static_cast<float>(get_current_time_ns() - record.timestamp) / 1e9f;
            return std::exp(-age / 10.0f);
        }
    }
    return 0.5f;
}

float VitalityOracle::compute_priority_bonus(const PackMetadata& pack) {
    float layer_bonus = 1.0f - static_cast<float>(pack.layer_idx) / 100.0f;
    return pack.base_priority * 0.5f + layer_bonus * 0.5f;
}

void VitalityOracle::save_model(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    file.write(reinterpret_cast<const char*>(model_.w1.data()), model_.w1.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(model_.b1.data()), model_.b1.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(model_.w2.data()), model_.w2.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(model_.b2.data()), model_.b2.size() * sizeof(float));
    file.close();
}

void VitalityOracle::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    file.read(reinterpret_cast<char*>(model_.w1.data()), model_.w1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(model_.b1.data()), model_.b1.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(model_.w2.data()), model_.w2.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(model_.b2.data()), model_.b2.size() * sizeof(float));
    file.close();
}

} // namespace vk_symbiote
