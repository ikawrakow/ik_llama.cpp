#include "chat.h"
#include "common.h"

#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

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
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & previous_msg, const common_chat_msg & new_msg) {
    std::vector<common_chat_msg_diff> diffs;
    if (previous_msg.reasoning_content != new_msg.reasoning_content) {
        auto & diff = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(previous_msg.reasoning_content, new_msg.reasoning_content);
    }
    if (previous_msg.content != new_msg.content) {
        auto & diff = diffs.emplace_back();
        diff.content_delta = string_diff(previous_msg.content, new_msg.content);
    }

    if (new_msg.tool_calls.size() < previous_msg.tool_calls.size()) {
        throw std::runtime_error("Invalid diff: now finding less tool calls!");
    }

    if (!previous_msg.tool_calls.empty()) {
        auto idx = previous_msg.tool_calls.size() - 1;
        const auto & pref = previous_msg.tool_calls[idx];
        const auto & newf = new_msg.tool_calls[idx];
        if (pref.name != newf.name) {
            throw std::runtime_error("Invalid diff: tool call mismatch!");
        }
        auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id) {
            auto & diff = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id) {
                diff.tool_call_delta.id = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = previous_msg.tool_calls.size(); idx < new_msg.tool_calls.size(); ++idx) {
        auto & diff = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = new_msg.tool_calls[idx];
    }
    return diffs;
}