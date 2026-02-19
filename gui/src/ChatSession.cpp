/**
 * ChatSession - Manages conversation state and context
 */

#include "../include/vk_symbiote_gui/SymbioteGUI.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <mutex>

namespace vk_symbiote {
namespace gui {

ChatSession::ChatSession() = default;

ChatSession::~ChatSession() = default;

void ChatSession::addMessage(const ChatMessage& message) {
    std::lock_guard<std::mutex> lock(messages_mutex_);
    messages_.push_back(message);
    
    // Update token count
    current_tokens_ += message.token_count;
    
    // Trim history if over limit
    while (current_tokens_ > max_context_tokens_ && messages_.size() > 1) {
        current_tokens_ -= messages_.front().token_count;
        messages_.erase(messages_.begin());
    }
}

std::vector<ChatMessage> ChatSession::getMessages() const {
    std::lock_guard<std::mutex> lock(messages_mutex_);
    return messages_;
}

void ChatSession::clear() {
    std::lock_guard<std::mutex> lock(messages_mutex_);
    messages_.clear();
    current_tokens_ = 0;
    context_segments_.clear();
}

uint32_t ChatSession::getTokenCount() const {
    return current_tokens_;
}

uint32_t ChatSession::getMaxTokens() const {
    return max_context_tokens_;
}

void ChatSession::setMaxTokens(uint32_t max_tokens) {
    max_context_tokens_ = max_tokens;
}

void ChatSession::addContextSegment(const std::string& label, uint32_t token_count) {
    ContextSegment segment;
    segment.label = label;
    segment.token_count = token_count;
    segment.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    context_segments_.push_back(segment);
    current_tokens_ += token_count;
}

std::vector<ChatSession::ContextSegment> ChatSession::getContextSegments() const {
    return context_segments_;
}

void ChatSession::loadContextFromFiles(const std::vector<std::string>& files) {
    for (const auto& file : files) {
        // Simple token estimation: ~4 chars per token
        size_t estimated_tokens = file.length() / 4;
        addContextSegment(file, static_cast<uint32_t>(estimated_tokens));
    }
}

std::string ChatSession::buildPrompt(const std::string& user_input) const {
    std::lock_guard<std::mutex> lock(messages_mutex_);
    
    std::string prompt;
    
    // Add system context if exists
    for (const auto& msg : messages_) {
        if (msg.type == ChatMessage::SYSTEM) {
            prompt += msg.content + "\n\n";
        }
    }
    
    // Add recent conversation
    size_t start_idx = 0;
    if (messages_.size() > 10) {
        start_idx = messages_.size() - 10;
    }
    
    for (size_t i = start_idx; i < messages_.size(); ++i) {
        const auto& msg = messages_[i];
        if (msg.type == ChatMessage::USER) {
            prompt += "User: " + msg.content + "\n";
        } else if (msg.type == ChatMessage::ASSISTANT) {
            prompt += "Assistant: " + msg.content + "\n";
        }
    }
    
    // Add current user input
    prompt += "User: " + user_input + "\nAssistant: ";
    
    return prompt;
}

} // namespace gui
} // namespace vk_symbiote
