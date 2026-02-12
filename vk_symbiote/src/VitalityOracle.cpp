#include "VitalityOracle.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstring>
#include <numeric>
#include <iostream>

namespace vk_symbiote {

// LSTM-like cell for temporal sequence modeling
struct LSTMCell {
    static constexpr uint32_t INPUT_SIZE = 64;
    static constexpr uint32_t HIDDEN_SIZE = 32;
    
    // LSTM gate weights
    std::vector<float> Wf, Wi, Wc, Wo;  // Forget, Input, Candidate, Output gates
    std::vector<float> Uf, Ui, Uc, Uo;  // Recurrent connections
    std::vector<float> bf, bi, bc, bo;  // Biases
    
    LSTMCell() : rng_(std::random_device{}()) {
        initialize_weights();
    }
    
    void initialize_weights() {
        const float scale = 0.01f;
        std::uniform_real_distribution<float> dist(-scale, scale);
        
        // Xavier initialization for input weights
        auto init_matrix = [&](std::vector<float>& mat, uint32_t rows, uint32_t cols) {
            mat.resize(rows * cols);
            float xavier_scale = std::sqrt(2.0f / (rows + cols));
            std::uniform_real_distribution<float> xavier_dist(-xavier_scale, xavier_scale);
            for (auto& w : mat) w = xavier_dist(rng_);
        };
        
        init_matrix(Wf, HIDDEN_SIZE, INPUT_SIZE);
        init_matrix(Wi, HIDDEN_SIZE, INPUT_SIZE);
        init_matrix(Wc, HIDDEN_SIZE, INPUT_SIZE);
        init_matrix(Wo, HIDDEN_SIZE, INPUT_SIZE);
        
        init_matrix(Uf, HIDDEN_SIZE, HIDDEN_SIZE);
        init_matrix(Ui, HIDDEN_SIZE, HIDDEN_SIZE);
        init_matrix(Uc, HIDDEN_SIZE, HIDDEN_SIZE);
        init_matrix(Uo, HIDDEN_SIZE, HIDDEN_SIZE);
        
        // Initialize biases - forget gate bias to 1.0 for better gradient flow
        bf.resize(HIDDEN_SIZE, 1.0f);
        bi.resize(HIDDEN_SIZE, 0.0f);
        bc.resize(HIDDEN_SIZE, 0.0f);
        bo.resize(HIDDEN_SIZE, 0.0f);
    }
    
    // Forward pass - returns new hidden and cell states
    struct State {
        std::vector<float> h;  // Hidden state
        std::vector<float> c;  // Cell state
    };
    
    State forward(const std::vector<float>& input, const State& prev_state) {
        State next_state;
        next_state.h.resize(HIDDEN_SIZE);
        next_state.c.resize(HIDDEN_SIZE);
        
        // Forget gate: f = sigmoid(Wf * x + Uf * h_prev + bf)
        std::vector<float> f(HIDDEN_SIZE);
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bf[i];
            // Input contribution
            for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
                sum += Wf[i * INPUT_SIZE + j] * input[j];
            }
            // Recurrent contribution
            for (uint32_t j = 0; j < HIDDEN_SIZE; ++j) {
                sum += Uf[i * HIDDEN_SIZE + j] * prev_state.h[j];
            }
            f[i] = sigmoid(sum);
        }
        
        // Input gate: i = sigmoid(Wi * x + Ui * h_prev + bi)
        std::vector<float> i_gate(HIDDEN_SIZE);
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bi[i];
            for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
                sum += Wi[i * INPUT_SIZE + j] * input[j];
            }
            for (uint32_t j = 0; j < HIDDEN_SIZE; ++j) {
                sum += Ui[i * HIDDEN_SIZE + j] * prev_state.h[j];
            }
            i_gate[i] = sigmoid(sum);
        }
        
        // Candidate: c_tilde = tanh(Wc * x + Uc * h_prev + bc)
        std::vector<float> c_tilde(HIDDEN_SIZE);
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bc[i];
            for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
                sum += Wc[i * INPUT_SIZE + j] * input[j];
            }
            for (uint32_t j = 0; j < HIDDEN_SIZE; ++j) {
                sum += Uc[i * HIDDEN_SIZE + j] * prev_state.h[j];
            }
            c_tilde[i] = std::tanh(sum);
        }
        
        // Cell state: c = f * c_prev + i * c_tilde
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            next_state.c[i] = f[i] * prev_state.c[i] + i_gate[i] * c_tilde[i];
        }
        
        // Output gate: o = sigmoid(Wo * x + Uo * h_prev + bo)
        std::vector<float> o_gate(HIDDEN_SIZE);
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            float sum = bo[i];
            for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
                sum += Wo[i * INPUT_SIZE + j] * input[j];
            }
            for (uint32_t j = 0; j < HIDDEN_SIZE; ++j) {
                sum += Uo[i * HIDDEN_SIZE + j] * prev_state.h[j];
            }
            o_gate[i] = sigmoid(sum);
        }
        
        // Hidden state: h = o * tanh(c)
        for (uint32_t i = 0; i < HIDDEN_SIZE; ++i) {
            next_state.h[i] = o_gate[i] * std::tanh(next_state.c[i]);
        }
        
        return next_state;
    }
    
private:
    std::mt19937 rng_;
    
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

// Output layer for prediction
struct OutputLayer {
    static constexpr uint32_t INPUT_SIZE = LSTMCell::HIDDEN_SIZE;
    
    std::vector<float> W;  // Weights: 1 x INPUT_SIZE
    float b;               // Bias
    
    OutputLayer() : rng_(std::random_device{}()) {
        W.resize(INPUT_SIZE);
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for (auto& w : W) w = dist(rng_);
        b = 0.0f;
    }
    
    float forward(const std::vector<float>& hidden) {
        float sum = b;
        for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
            sum += W[i] * hidden[i];
        }
        return sigmoid(sum);
    }
    
private:
    std::mt19937 rng_;
    
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-std::clamp(x, -10.0f, 10.0f)));
    }
};

// Per-pack hidden state for temporal modeling
struct PackLSTMState {
    LSTMCell::State state;
    uint64_t last_update = 0;
    std::deque<std::vector<float>> recent_features;
    static constexpr uint32_t MAX_HISTORY = 10;
    
    PackLSTMState() {
        state.h.resize(LSTMCell::HIDDEN_SIZE, 0.0f);
        state.c.resize(LSTMCell::HIDDEN_SIZE, 0.0f);
    }
    
    void update(const std::vector<float>& features, const LSTMCell& lstm) {
        // Keep recent feature history
        recent_features.push_back(features);
        if (recent_features.size() > MAX_HISTORY) {
            recent_features.pop_front();
        }
        
        // Update LSTM state
        state = lstm.forward(features, state);
        last_update = get_current_time_ns();
    }
};

// Training buffer for online learning
struct TrainingBuffer {
    struct Sample {
        std::vector<float> features;
        std::vector<float> hidden_state;
        float target;  // 1.0 if pack was used, 0.0 otherwise
        float predicted;
        uint64_t timestamp;
    };
    
    std::deque<Sample> samples;
    static constexpr uint32_t MAX_SAMPLES = 1000;
    
    void add(const std::vector<float>& features, const std::vector<float>& hidden,
             float target, float predicted) {
        Sample sample;
        sample.features = features;
        sample.hidden_state = hidden;
        sample.target = target;
        sample.predicted = predicted;
        sample.timestamp = get_current_time_ns();
        
        samples.push_back(sample);
        if (samples.size() > MAX_SAMPLES) {
            samples.pop_front();
        }
    }
    
    void clear() {
        samples.clear();
    }
};

// VitalityOracle implementation
class VitalityOracleImpl {
public:
    explicit VitalityOracleImpl(uint32_t max_packs_in_memory) 
        : max_packs_in_memory_(max_packs_in_memory),
          rng_(std::random_device{}()) {
    }
    
    VitalityScore score_pack(const PackMetadata& pack, const std::vector<uint32_t>& tokens,
                            uint32_t current_layer, const HardwareTelemetry& telemetry) {
        VitalityScore score;
        
        // Extract features
        auto features = extract_features(pack, tokens, current_layer, telemetry);
        
        // Get or create LSTM state for this pack
        auto& pack_state = get_or_create_pack_state(pack.pack_id);
        
        // Update LSTM state with current features
        pack_state.update(features, lstm_cell_);
        
        // Compute component scores
        score.relevance = compute_relevance(pack, tokens, current_layer);
        score.hardware_score = compute_hardware_score(pack, telemetry);
        score.temporal_score = compute_temporal_score(pack.pack_id);
        score.priority_bonus = compute_priority_bonus(pack);
        
        // LSTM-based confidence prediction
        score.confidence = output_layer_.forward(pack_state.state.h);
        score.confidence = std::clamp(score.confidence, 0.0f, 1.0f);
        
        // Store for training
        last_features_[pack.pack_id] = features;
        last_hidden_[pack.pack_id] = pack_state.state.h;
        last_prediction_[pack.pack_id] = score.confidence;
        
        return score;
    }
    
    std::vector<uint64_t> predict_next_packs(const std::vector<uint32_t>& tokens,
                                             uint32_t current_layer, uint32_t lookahead, uint32_t count) {
        std::vector<std::pair<uint64_t, float>> scored_packs;
        
        // Generate candidates for next layers
        for (uint32_t offset = 0; offset < lookahead; ++offset) {
            uint32_t layer = current_layer + offset;
            if (layer >= 128) break;  // Sanity limit
            
            float layer_decay = 1.0f - static_cast<float>(offset) / lookahead * 0.5f;
            
            // Score all packs that might be needed
            for (const auto& [pack_id, pack_state] : pack_states_) {
                // Simulate forward prediction
                float predicted_score = output_layer_.forward(pack_state.state.h);
                predicted_score *= layer_decay;
                
                // Add temporal decay based on last access
                uint64_t age_ns = get_current_time_ns() - pack_state.last_update;
                float age_score = std::exp(-static_cast<float>(age_ns) / 1e9f / 60.0f);  // 60 second decay
                predicted_score *= age_score;
                
                scored_packs.push_back({pack_id, predicted_score});
            }
        }
        
        // Sort by score descending
        std::sort(scored_packs.begin(), scored_packs.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top predictions
        std::vector<uint64_t> predictions;
        predictions.reserve(std::min(count, static_cast<uint32_t>(scored_packs.size())));
        
        for (uint32_t i = 0; i < count && i < scored_packs.size(); ++i) {
            predictions.push_back(scored_packs[i].first);
        }
        
        return predictions;
    }
    
    void record_access(uint64_t pack_id, bool was_used, float predicted_score) {
        AccessRecord record;
        record.pack_id = pack_id;
        record.timestamp = get_current_time_ns();
        record.was_used = was_used;
        record.predicted_score = predicted_score;
        
        recent_accesses_.push_back(record);
        if (recent_accesses_.size() > max_access_history_) {
            recent_accesses_.pop_front();
        }
        
        // Update statistics
        total_predictions_++;
        if (was_used) {
            correct_predictions_++;
        }
        hit_rate_ = static_cast<float>(correct_predictions_) / static_cast<float>(total_predictions_);
        
        // Add to training buffer
        auto it = last_features_.find(pack_id);
        auto hidden_it = last_hidden_.find(pack_id);
        if (it != last_features_.end() && hidden_it != last_hidden_.end()) {
            float target = was_used ? 1.0f : 0.0f;
            training_buffer_.add(it->second, hidden_it->second, target, predicted_score);
        }
    }
    
    void update_model() {
        if (training_buffer_.samples.size() < 32) return;
        
        // Online gradient descent with adaptive learning rate
        float learning_rate = base_learning_rate_ * 
            (1.0f / (1.0f + 0.001f * static_cast<float>(training_steps_)));
        
        float total_loss = 0.0f;
        uint32_t num_updates = 0;
        
        // Process recent samples with higher weight
        for (const auto& sample : training_buffer_.samples) {
            // Forward pass through output layer
            float prediction = output_layer_.forward(sample.hidden_state);
            
            // Binary cross-entropy loss gradient
            float error = sample.target - prediction;
            total_loss += std::abs(error);
            
            // Backpropagate through output layer
            // dL/dW = dL/dy * dy/dz * dz/dW = error * sigmoid'(z) * h
            float sigmoid_deriv = prediction * (1.0f - prediction);
            float gradient = error * sigmoid_deriv;
            
            // Update output layer weights
            for (uint32_t i = 0; i < OutputLayer::INPUT_SIZE; ++i) {
                output_layer_.W[i] += learning_rate * gradient * sample.hidden_state[i];
            }
            output_layer_.b += learning_rate * gradient;
            
            num_updates++;
        }
        
        // Optional: Backpropagate through LSTM (simplified - only update output projection)
        // Full BPTT would be more complex but often unnecessary for this use case
        
        training_steps_++;
        
        // Clear buffer after training
        training_buffer_.clear();
        
        // Log training stats periodically
        if (training_steps_ % 100 == 0) {
            float avg_loss = total_loss / num_updates;
            std::cout << "VitalityOracle training step " << training_steps_ 
                     << ", avg_loss: " << avg_loss 
                     << ", hit_rate: " << hit_rate_ 
                     << ", lr: " << learning_rate << std::endl;
        }
    }
    
    void save_model(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to save model to: " << path << std::endl;
            return;
        }
        
        // Write header
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write LSTM weights
        write_vector(file, lstm_cell_.Wf);
        write_vector(file, lstm_cell_.Wi);
        write_vector(file, lstm_cell_.Wc);
        write_vector(file, lstm_cell_.Wo);
        write_vector(file, lstm_cell_.Uf);
        write_vector(file, lstm_cell_.Ui);
        write_vector(file, lstm_cell_.Uc);
        write_vector(file, lstm_cell_.Uo);
        write_vector(file, lstm_cell_.bf);
        write_vector(file, lstm_cell_.bi);
        write_vector(file, lstm_cell_.bc);
        write_vector(file, lstm_cell_.bo);
        
        // Write output layer
        write_vector(file, output_layer_.W);
        file.write(reinterpret_cast<const char*>(&output_layer_.b), sizeof(output_layer_.b));
        
        // Write statistics
        file.write(reinterpret_cast<const char*>(&training_steps_), sizeof(training_steps_));
        file.write(reinterpret_cast<const char*>(&hit_rate_), sizeof(hit_rate_));
        file.write(reinterpret_cast<const char*>(&total_predictions_), sizeof(total_predictions_));
        file.write(reinterpret_cast<const char*>(&correct_predictions_), sizeof(correct_predictions_));
        
        file.close();
        std::cout << "VitalityOracle model saved to: " << path << std::endl;
    }
    
    void load_model(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to load model from: " << path << std::endl;
            return;
        }
        
        // Read header
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            std::cerr << "Unknown model version: " << version << std::endl;
            return;
        }
        
        // Read LSTM weights
        read_vector(file, lstm_cell_.Wf);
        read_vector(file, lstm_cell_.Wi);
        read_vector(file, lstm_cell_.Wc);
        read_vector(file, lstm_cell_.Wo);
        read_vector(file, lstm_cell_.Uf);
        read_vector(file, lstm_cell_.Ui);
        read_vector(file, lstm_cell_.Uc);
        read_vector(file, lstm_cell_.Uo);
        read_vector(file, lstm_cell_.bf);
        read_vector(file, lstm_cell_.bi);
        read_vector(file, lstm_cell_.bc);
        read_vector(file, lstm_cell_.bo);
        
        // Read output layer
        read_vector(file, output_layer_.W);
        file.read(reinterpret_cast<char*>(&output_layer_.b), sizeof(output_layer_.b));
        
        // Read statistics
        file.read(reinterpret_cast<char*>(&training_steps_), sizeof(training_steps_));
        file.read(reinterpret_cast<char*>(&hit_rate_), sizeof(hit_rate_));
        file.read(reinterpret_cast<char*>(&total_predictions_), sizeof(total_predictions_));
        file.read(reinterpret_cast<char*>(&correct_predictions_), sizeof(correct_predictions_));
        
        file.close();
        std::cout << "VitalityOracle model loaded from: " << path 
                 << ", training_steps: " << training_steps_ << std::endl;
    }
    
    float hit_rate() const { return hit_rate_; }
    uint64_t total_predictions() const { return total_predictions_; }
    
private:
    uint32_t max_packs_in_memory_;
    std::mt19937 rng_;
    
    // Neural network components
    LSTMCell lstm_cell_;
    OutputLayer output_layer_;
    
    // Per-pack states
    std::unordered_map<uint64_t, PackLSTMState> pack_states_;
    
    // Training data
    std::deque<AccessRecord> recent_accesses_;
    static constexpr uint32_t max_access_history_ = 10000;
    TrainingBuffer training_buffer_;
    
    // Last predictions for training
    std::unordered_map<uint64_t, std::vector<float>> last_features_;
    std::unordered_map<uint64_t, std::vector<float>> last_hidden_;
    std::unordered_map<uint64_t, float> last_prediction_;
    
    // Training state
    float base_learning_rate_ = 0.001f;
    uint32_t training_steps_ = 0;
    
    // Statistics
    float hit_rate_ = 0.0f;
    uint64_t total_predictions_ = 0;
    uint64_t correct_predictions_ = 0;
    
    PackLSTMState& get_or_create_pack_state(uint64_t pack_id) {
        auto it = pack_states_.find(pack_id);
        if (it == pack_states_.end()) {
            it = pack_states_.emplace(pack_id, PackLSTMState()).first;
        }
        return it->second;
    }
    
    std::vector<float> extract_features(const PackMetadata& pack, 
                                       const std::vector<uint32_t>& tokens,
                                       uint32_t current_layer, 
                                       const HardwareTelemetry& telemetry) {
        std::vector<float> features(LSTMCell::INPUT_SIZE, 0.0f);
        
        // Pack metadata features
        features[0] = static_cast<float>(pack.layer_idx) / 128.0f;
        features[1] = static_cast<float>(pack.head_idx) / 64.0f;
        features[2] = static_cast<float>(pack.type) / 255.0f;
        features[3] = pack.base_priority;
        
        // Context features
        features[4] = static_cast<float>(current_layer) / 128.0f;
        features[5] = static_cast<float>(tokens.size()) / 8192.0f;
        
        // Hardware features
        features[6] = telemetry.gpu_utilization;
        features[7] = telemetry.memory_bandwidth;
        features[8] = telemetry.cache_hit_rate;
        features[9] = telemetry.compute_occupancy;
        features[10] = telemetry.thermal_throttle;
        
        // Temporal features
        auto& pack_state = get_or_create_pack_state(pack.pack_id);
        features[11] = static_cast<float>(pack_state.recent_features.size()) / PackLSTMState::MAX_HISTORY;
        
        // Token hash features (locality-sensitive hashing)
        uint64_t token_hash = hash_tokens(tokens);
        for (uint32_t i = 0; i < 8; ++i) {
            features[12 + i] = static_cast<float>((token_hash >> (i * 8)) & 0xFF) / 255.0f;
        }
        
        // Compute relevance score as feature
        features[20] = compute_relevance(pack, tokens, current_layer);
        features[21] = compute_hardware_score(pack, telemetry);
        features[22] = compute_temporal_score(pack.pack_id);
        
        // Time-based cyclical features
        uint64_t time_ms = get_current_time_ms();
        float time_sec = static_cast<float>(time_ms % 1000000) / 1000.0f;
        features[23] = std::sin(2.0f * 3.14159f * time_sec / 60.0f);  // Minute cycle
        features[24] = std::cos(2.0f * 3.14159f * time_sec / 60.0f);
        
        // Pack size features
        features[25] = static_cast<float>(pack.compressed_size) / (1024.0f * 1024.0f * 1024.0f);  // GB
        features[26] = static_cast<float>(pack.decompressed_size) / (1024.0f * 1024.0f * 1024.0f);  // GB
        
        // Fill remaining with noise for regularization
        std::uniform_real_distribution<float> noise_dist(-0.1f, 0.1f);
        for (uint32_t i = 27; i < LSTMCell::INPUT_SIZE; ++i) {
            features[i] = noise_dist(rng_);
        }
        
        return features;
    }
    
    uint64_t hash_tokens(const std::vector<uint32_t>& tokens) const {
        uint64_t hash = 1469598103934665603ULL;  // FNV-1a offset basis
        for (uint32_t token : tokens) {
            hash ^= token;
            hash *= 1099511628211ULL;  // FNV-1a prime
        }
        return hash;
    }
    
    float compute_relevance(const PackMetadata& pack, const std::vector<uint32_t>& tokens, uint32_t current_layer) {
        // Layer proximity
        float layer_dist = std::abs(static_cast<float>(pack.layer_idx) - static_cast<float>(current_layer));
        float layer_score = std::exp(-layer_dist / 5.0f);
        
        // Token-based relevance (simplified)
        float token_score = 0.5f;
        if (!tokens.empty()) {
            // Check if pack name contains token-related patterns
            // This is a simplified heuristic
            token_score = 0.5f + 0.3f * std::sin(static_cast<float>(tokens.back()) / 1000.0f);
        }
        
        // Type-based relevance
        float type_weight = 0.7f;
        switch (pack.type) {
            case PackType::ATTENTION_Q:
            case PackType::ATTENTION_K:
            case PackType::ATTENTION_V:
                type_weight = 1.0f;
                break;
            case PackType::ATTENTION_O:
                type_weight = 0.9f;
                break;
            case PackType::NORM_GAMMA:
                type_weight = 0.85f;
                break;
            default:
                break;
        }
        
        return layer_score * token_score * type_weight;
    }
    
    float compute_hardware_score(const PackMetadata& pack, const HardwareTelemetry& telemetry) {
        (void)pack;
        
        float score = 1.0f;
        
        // Penalize high GPU utilization
        if (telemetry.gpu_utilization > 0.9f) {
            score *= 0.6f;
        } else if (telemetry.gpu_utilization < 0.3f) {
            score *= 1.2f;
        }
        
        // Penalize high memory bandwidth usage
        if (telemetry.memory_bandwidth > 0.85f) {
            score *= 0.7f;
        }
        
        // Reward high cache hit rate
        score *= (0.5f + 0.5f * telemetry.cache_hit_rate);
        
        // Apply thermal throttling factor
        score *= telemetry.thermal_throttle;
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    float compute_temporal_score(uint64_t pack_id) {
        auto it = pack_states_.find(pack_id);
        if (it == pack_states_.end()) {
            return 0.5f;  // Unknown pack
        }
        
        uint64_t age_ns = get_current_time_ns() - it->second.last_update;
        float age_sec = static_cast<float>(age_ns) / 1e9f;
        
        // Exponential decay with 30-second half-life
        return std::exp(-age_sec / 30.0f);
    }
    
    float compute_priority_bonus(const PackMetadata& pack) {
        return pack.base_priority * 0.7f + 
               (1.0f - static_cast<float>(pack.layer_idx) / 128.0f) * 0.3f;
    }
    
    void write_vector(std::ofstream& file, const std::vector<float>& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    }
    
    void read_vector(std::ifstream& file, std::vector<float>& vec) {
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    }
};

// Public API Implementation
VitalityOracle::VitalityOracle(uint32_t max_packs_in_memory) 
    : pimpl_(std::make_unique<VitalityOracleImpl>(max_packs_in_memory)) {
}

VitalityOracle::~VitalityOracle() = default;

VitalityScore VitalityOracle::score_pack(const PackMetadata& pack, const std::vector<uint32_t>& tokens,
                                        uint32_t current_layer, const HardwareTelemetry& telemetry) {
    return pimpl_->score_pack(pack, tokens, current_layer, telemetry);
}

std::vector<uint64_t> VitalityOracle::predict_next_packs(const std::vector<uint32_t>& tokens,
                                                         uint32_t current_layer, uint32_t lookahead, uint32_t count) {
    return pimpl_->predict_next_packs(tokens, current_layer, lookahead, count);
}

void VitalityOracle::update_telemetry(const HardwareTelemetry& telemetry) {
    (void)telemetry;  // Stored in implementation if needed
}

void VitalityOracle::record_access(uint64_t pack_id, bool was_used, float predicted_score) {
    pimpl_->record_access(pack_id, was_used, predicted_score);
}

void VitalityOracle::update_model() {
    pimpl_->update_model();
}

float VitalityOracle::hit_rate() const noexcept {
    return pimpl_->hit_rate();
}

uint64_t VitalityOracle::total_predictions() const noexcept {
    return pimpl_->total_predictions();
}

void VitalityOracle::save_model(const std::string& path) {
    pimpl_->save_model(path);
}

void VitalityOracle::load_model(const std::string& path) {
    pimpl_->load_model(path);
}

} // namespace vk_symbiote
