#pragma once

#include "Common.h"
#include "NomadPack.h"
#include <random>
#include <deque>

namespace vk_symbiote {

struct VitalityScore {
    float relevance = 0.0f, hardware_score = 0.0f, temporal_score = 0.0f;
    float priority_bonus = 0.0f, confidence = 0.0f;
    float total() const noexcept { return relevance + hardware_score + temporal_score + priority_bonus; }
};

struct HardwareTelemetry {
    float gpu_utilization = 0.0f, memory_bandwidth = 0.0f, cache_hit_rate = 0.0f;
    float compute_occupancy = 0.0f, thermal_throttle = 1.0f;
    uint64 timestamp = 0;
};

struct AccessRecord {
    uint64 pack_id = 0, timestamp = 0;
    bool was_used = false;
    float predicted_score = 0.0f;
};

struct MLPModel {
    std::vector<float> w1; std::vector<float> b1;
    std::vector<float> w2; std::vector<float> b2;
    uint32 input_size = 64, hidden_size = 16;
};

class VitalityOracle {
public:
    explicit VitalityOracle(uint32 max_packs_in_memory = 64);
    ~VitalityOracle();

    VitalityScore score_pack(const PackMetadata& pack, const std::vector<uint32>& tokens,
                             uint32 current_layer, const HardwareTelemetry& telemetry);

    std::vector<uint64> predict_next_packs(const std::vector<uint32>& tokens,
                                           uint32 current_layer, uint32 lookahead = 3, uint32 count = 8);

    void update_telemetry(const HardwareTelemetry& telemetry);
    void record_access(uint64 pack_id, bool was_used, float predicted_score);
    void update_model();

    void set_weights(float relevance, float hardware, float temporal, float priority) {
        weight_relevance_ = relevance; weight_hardware_ = hardware;
        weight_temporal_ = temporal; weight_priority_ = priority;
    }

    float hit_rate() const noexcept { return hit_rate_; }
    uint64 total_predictions() const noexcept { return total_predictions_; }

    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    MLPModel model_;
    std::mt19937 rng_{std::random_device{}()};
    std::deque<AccessRecord> recent_accesses_;
    uint64_t max_access_history_ = 10000;

    float weight_relevance_ = 0.3f, weight_hardware_ = 0.25f;
    float weight_temporal_ = 0.2f, weight_priority_ = 0.25f;

    float hit_rate_ = 0.0f;
    uint64_t total_predictions_ = 0, correct_predictions_ = 0;

    std::vector<float> extract_features(const PackMetadata& pack, const std::vector<uint32>& tokens,
                                       uint32 current_layer, const HardwareTelemetry& telemetry);
    float mlp_forward(const std::vector<float>& input);
    void mlp_backward(const std::vector<float>& input, const std::vector<float>& gradient, float lr);

    uint64_t hash_tokens(const std::vector<uint32>& tokens) const;
    float compute_relevance(const PackMetadata& pack, const std::vector<uint32>& tokens, uint32 current_layer);
    float compute_hardware_score(const PackMetadata& pack, const HardwareTelemetry& telemetry);
    float compute_temporal_score(uint64 pack_id);
    float compute_priority_bonus(const PackMetadata& pack);
};

} // namespace vk_symbiote
