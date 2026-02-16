#pragma once

#include "Common.h"
#include "NomadPack.h"
#include <random>
#include <deque>

namespace vk_symbiote {

// Forward declaration for PIMPL idiom
class VitalityOracleImpl;

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

struct LSTMWeightStats {
    float mean = 0.0f, std = 0.0f, min = 0.0f, max = 0.0f;
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

    float hit_rate() const noexcept;
    uint64 total_predictions() const noexcept;
    uint32 get_training_steps() const;

    // Model persistence with TOML format
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    
    // Entropy-based code detection
    float calculate_token_entropy(const std::vector<uint32>& tokens);
    bool is_code_prompt(const std::vector<uint32>& tokens);
    
    // SGD training configuration
    void set_learning_rate(float lr);
    float get_learning_rate() const;
    void enable_sgd_training(bool enable);
    
    // Adam optimizer support
    void enable_adam_training(bool enable);
    bool is_adam_enabled() const;
    
    // Momentum configuration
    void set_momentum(float m);
    float get_momentum() const;
    
    // LSTM diagnostics
    void print_lstm_stats(uint64 pack_id);

private:
    std::unique_ptr<VitalityOracleImpl> pimpl_;
    
    // Weights for backward compatibility
    float weight_relevance_ = 0.3f, weight_hardware_ = 0.25f;
    float weight_temporal_ = 0.2f, weight_priority_ = 0.25f;
};

} // namespace vk_symbiote
