#include "chat.h"
#include "chat-parser.h"
#include "common.h"
#include "../examples/server/parsers/kimi_k2_parser.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include "json.hpp"

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

// Format parsing functions (ported from original llama.cpp)
// Content-only parsing (internal implementation - matches llama.cpp exactly)
static void common_chat_parse_content_only(common_chat_msg_parser & builder) {
    builder.add_content(builder.consume_rest());
}

static void common_chat_parse_generic(common_chat_msg_parser & builder) {
    if (!builder.syntax().enable_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const std::vector<std::vector<std::string>> content_paths = {
        {"response"},
    };
    static const std::vector<std::vector<std::string>> args_paths = {
        {"tool_call", "arguments"},
        {"tool_calls", "arguments"},
    };
    auto data = builder.consume_json_with_dumped_args(args_paths, content_paths);
    if (data.value.contains("tool_calls")) {
        if (!builder.add_tool_calls(data.value.at("tool_calls")) || data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool calls");
        }
    } else if (data.value.contains("tool_call")) {
        if (!builder.add_tool_call(data.value.at("tool_call")) || data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
    } else if (data.value.contains("response")) {
        const auto & response = data.value.at("response");
        builder.add_content(response.is_string() ? response.template get<std::string>() : response.dump(2));
        if (data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete response");
        }
    } else {
        throw common_chat_msg_partial_exception("Expected 'tool_call', 'tool_calls' or 'response' in JSON");
    }
}

static void common_chat_parse_deepseek_r1(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().enable_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?function<｜tool▁sep｜>([^\n]+)\n```json\n");
    static const common_regex close_regex("```[\\s\\r\\n]*<｜tool▁call▁end｜>");

    // Simplified tool calls parsing for DEEPSEEK_R1
    if (auto res = builder.try_find_regex(tool_calls_begin)) {
        while (auto func_res = builder.try_find_regex(function_regex)) {
            auto function_name = builder.str(func_res->groups[1]);
            auto args_json = builder.try_consume_json();
            if (args_json) {
                builder.add_tool_call(function_name, "", args_json->json.dump());
                builder.try_consume_regex(close_regex);
            } else {
                throw common_chat_msg_partial_exception("incomplete tool call JSON");
            }
        }
        builder.try_consume_regex(tool_calls_end);
        builder.add_content(builder.consume_rest());
    } else {
        builder.add_content(builder.consume_rest());
    }
}

static void common_chat_parse_kimi_k2(common_chat_msg_parser & builder) {
    // Delegate to existing Kimi-K2 implementation for backward compatibility
    auto result = kimi_k2::parse_tool_calls(builder.input());
    for (const auto& tc_json : result) {
        common_chat_tool_call tc;
        tc.id = tc_json.value("id", "");
        if (tc_json.contains("function") && tc_json["function"].contains("name")) {
            tc.name = tc_json["function"]["name"];
            tc.arguments = tc_json["function"].value("arguments", "{}");
            builder.add_tool_call(tc);
        }
    }
    // Add cleaned content (removes tool call syntax)
    builder.add_content(kimi_k2::clean_content(builder.input()));
}

// Main parsing dispatch function
static void common_chat_parse(common_chat_msg_parser & builder) {
    switch (builder.syntax().format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            common_chat_parse_content_only(builder);
            break;
        case COMMON_CHAT_FORMAT_GENERIC:
            common_chat_parse_generic(builder);
            break;
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:
            common_chat_parse_deepseek_r1(builder);
            break;
        case COMMON_CHAT_FORMAT_KIMI_K2:
            common_chat_parse_kimi_k2(builder);
            break;
        default:
            throw std::runtime_error(std::string("Unsupported format: ") + common_chat_format_name(builder.syntax().format));
    }
    builder.finish();
}

// Main public parsing function
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax) {
    common_chat_msg_parser builder(input, is_partial, syntax);
    try {
        common_chat_parse(builder);
    } catch (const common_chat_msg_partial_exception & ex) {
        if (!is_partial) {
            // Fallback to content-only on parsing errors
            builder.clear_tools();
            builder.move_to(0);
            common_chat_parse_content_only(builder);
        }
        // Re-throw for partial cases to signal incomplete parsing
        if (is_partial) {
            throw;
        }
    }
    return builder.result();
}

// Get format name for debugging/logging
const char* common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY: return "content_only";
        case COMMON_CHAT_FORMAT_GENERIC:      return "generic";
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:  return "deepseek_r1";
        case COMMON_CHAT_FORMAT_KIMI_K2:      return "kimi_k2";
        default:                              return "unknown";
    }
}