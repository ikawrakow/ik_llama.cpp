#pragma once

#include "../../common/common.h"
#include "json.hpp"
#include <string>
#include <vector>
#include <functional>

using json = nlohmann::ordered_json;

//
// Streaming chat data structures ported from original llama.cpp
// Enables differential streaming of tool calls during generation
//

// Tool call structure for streaming
struct ik_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const ik_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }

    bool operator!=(const ik_chat_tool_call & other) const {
        return !(*this == other);
    }
};

// Chat message structure with tool call support
struct ik_chat_msg {
    std::string role;
    std::string content;
    std::vector<ik_chat_tool_call> tool_calls = {};

    // Check if message is empty
    bool empty() const {
        return content.empty() && tool_calls.empty();
    }
    
    // Ensure all tool calls have IDs set
    void ensure_tool_call_ids_set(std::vector<std::string> & ids_cache, const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }
    
    bool operator==(const ik_chat_msg & other) const {
        return role == other.role
            && content == other.content
            && tool_calls == other.tool_calls;
    }
    
    bool operator!=(const ik_chat_msg & other) const {
        return !(*this == other);
    }
};

// Differential update structure for streaming
struct ik_chat_msg_diff {
    std::string content_delta;
    size_t tool_call_index = std::string::npos;
    ik_chat_tool_call tool_call_delta;

    // Compute differences between two messages for streaming
    static std::vector<ik_chat_msg_diff> compute_diffs(const ik_chat_msg & previous_msg, const ik_chat_msg & new_msg);

    bool operator==(const ik_chat_msg_diff & other) const {
        return content_delta == other.content_delta
            && tool_call_index == other.tool_call_index
            && tool_call_delta == other.tool_call_delta;
    }
};



// Helper functions for string diffing
static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        // For robustness, return the full current string if diff fails
        return current;
    }
    return current.substr(last.size());
}

// Implementation of compute_diffs function
inline std::vector<ik_chat_msg_diff> ik_chat_msg_diff::compute_diffs(const ik_chat_msg & previous_msg, const ik_chat_msg & new_msg) {
    std::vector<ik_chat_msg_diff> diffs;
    
    // Compute content diff
    if (previous_msg.content != new_msg.content) {
        auto & diff = diffs.emplace_back();
        diff.content_delta = string_diff(previous_msg.content, new_msg.content);
    }

    // Validate tool call consistency
    if (new_msg.tool_calls.size() < previous_msg.tool_calls.size()) {
        // For robustness, handle this case by treating as content change
        // Rather than throwing an exception
        return diffs;
    }

    // Compute diff for existing tool calls (arguments may be extended)
    if (!previous_msg.tool_calls.empty() && !new_msg.tool_calls.empty()) {
        auto idx = previous_msg.tool_calls.size() - 1;
        
        // Safety check: ensure index is valid for new message
        if (idx < new_msg.tool_calls.size()) {
            const auto & prev_call = previous_msg.tool_calls[idx];
            const auto & new_call = new_msg.tool_calls[idx];
            
            // Check if this is the same tool call being extended
            if (prev_call.name == new_call.name || new_call.name.empty()) {
                try {
                    auto args_diff = string_diff(prev_call.arguments, new_call.arguments);
                    if (!args_diff.empty() || prev_call.id != new_call.id) {
                        auto & diff = diffs.emplace_back();
                        diff.tool_call_index = idx;
                        if (prev_call.id != new_call.id) {
                            diff.tool_call_delta.id = new_call.id;
                            diff.tool_call_delta.name = new_call.name;
                        }
                        diff.tool_call_delta.arguments = args_diff;
                    }
                } catch (const std::exception&) {
                    // Skip if string diff fails
                }
            }
        }
    }
    
    // Add new tool calls
    for (size_t idx = previous_msg.tool_calls.size(); idx < new_msg.tool_calls.size(); ++idx) {
        auto & diff = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = new_msg.tool_calls[idx];
    }
    
    return diffs;
}

// Convert diff to OpenAI streaming format
static json chat_msg_diff_to_oai_streaming(const ik_chat_msg_diff & diff) {
    json delta = json::object();
    
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        
        json function = json::object();
        if (!diff.tool_call_delta.name.empty()) {
            function["name"] = diff.tool_call_delta.name;
        }
        function["arguments"] = diff.tool_call_delta.arguments;
        tool_call["function"] = function;
        
        delta["tool_calls"] = json::array({tool_call});
    }
    
    return delta;
}

// Generate streaming chunks from diffs
static std::vector<json> generate_streaming_chunks(const std::vector<ik_chat_msg_diff> & diffs, const std::string & completion_id, const std::string & model_name) {
    std::vector<json> chunks;
    std::time_t t = std::time(0);
    
    for (const auto & diff : diffs) {
        try {
            json delta = chat_msg_diff_to_oai_streaming(diff);
            if (!delta.empty()) {
                json chunk = {
                    {"choices", json::array({json{
                        {"finish_reason", nullptr},
                        {"index", 0},
                        {"delta", delta}
                    }})},
                    {"created", t},
                    {"id", completion_id},
                    {"model", model_name},
                    {"object", "chat.completion.chunk"}
                };
                chunks.push_back(chunk);
            }
        } catch (const std::exception&) {
            // Skip malformed diffs but continue processing
            continue;
        }
    }
    
    return chunks;
}