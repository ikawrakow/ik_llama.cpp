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

// Helper function from original llama.cpp
static std::string wrap_code_as_arguments(common_chat_msg_parser & builder, const std::string & code) {
    std::string arguments;
    if (builder.is_partial()) {
        arguments = (json {{"code", code + builder.healing_marker()}}).dump();
        auto idx = arguments.find(builder.healing_marker());
        if (idx != std::string::npos) {
            arguments.resize(idx);
        }
    } else {
        arguments = (json {{"code", code}}).dump();
    }
    return arguments;
}

// Forward declaration
static void parse_deepseek_r1_tools_array(common_chat_msg_parser & builder);
static void parse_deepseek_r1_xml_wrapped(common_chat_msg_parser & builder);

// Helper function from original llama.cpp for parsing JSON tool calls
static void parse_json_tool_calls(
    common_chat_msg_parser & builder,
    const std::optional<common_regex> & block_open,
    const std::optional<common_regex> & function_regex_start_only,
    const std::optional<common_regex> & function_regex,
    const common_regex & close_regex,
    const std::optional<common_regex> & block_close,
    bool allow_raw_python = false,
    const std::function<std::string(const common_chat_msg_parser::find_regex_result & fres)> & get_function_name = nullptr) {

    auto parse_tool_calls = [&]() {
        size_t from = std::string::npos;
        auto first = true;
        while (true) {
            auto res = function_regex_start_only && first
                ? builder.try_consume_regex(*function_regex_start_only)
                : function_regex
                    ? builder.try_find_regex(*function_regex, from)
                    : std::nullopt;
            if (res) {
                std::string name;
                if (get_function_name) {
                    name = get_function_name(*res);
                } else {
                    if (res->groups.size() < 2) {
                        from = res->groups[0].begin + 1;
                        continue;
                    }
                    name = builder.str(res->groups[1]);
                }
                first = false;
                if (name.empty()) {
                    // get_function_name signalled us that we should skip this match and treat it as content.
                    from = res->groups[0].begin + 1;
                    continue;
                }
                from = std::string::npos;

                auto maybe_raw_python = name == "python" && allow_raw_python;
                if (builder.input()[builder.pos()] == '{' || !maybe_raw_python) {
                    if (auto arguments = builder.try_consume_json_with_dumped_args({{}})) {
                        if (!builder.add_tool_call(name, "", arguments->value) || arguments->is_partial) {
                            throw common_chat_msg_partial_exception("incomplete tool call");
                        }
                        builder.try_consume_regex(close_regex);
                    }
                    continue;
                }
                if (maybe_raw_python) {
                    auto arguments = wrap_code_as_arguments(builder, builder.consume_rest());
                    if (!builder.add_tool_call(name, "", arguments)) {
                        throw common_chat_msg_partial_exception("incomplete tool call");
                    }
                    return;
                }
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
            break;
        }
        if (block_close) {
            builder.try_consume_regex(*block_close);
        }
        builder.consume_spaces();
        builder.add_content(builder.consume_rest());
    };
    if (block_open) {
        if (auto res = builder.try_find_regex(*block_open)) {
            parse_tool_calls();
        } else {
            builder.add_content(builder.consume_rest());
        }
    } else {
        parse_tool_calls();
    }
}

void common_chat_parse_deepseek_r1(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().enable_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");
    // Primary regex for correct format with separator
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?function<｜tool▁sep｜>([^\n]+)\n```json\n");
    // Fallback regex for format without separator (some models generate this)
    static const common_regex function_regex_no_sep("(?:<｜tool▁call▁begin｜>)?function<([^>]+)>\n```json\n");
    // Third regex for new format: just "function" with no markers
    static const common_regex function_regex_simple("function\n```json\n");
    static const common_regex close_regex("```[\\s\\r\\n]*<｜tool▁call▁end｜>");
    static const common_regex close_regex_simple("```");  // For simple format without end markers

    // Check for the new tools array format first (no DeepSeek markers)
    auto original_pos = builder.pos();

    // First, try the tools array format for content like "function\n```json\n{"tools": [...]}"
    if (builder.try_find_regex(function_regex_simple)) {
        builder.move_to(original_pos);
        try {
            parse_deepseek_r1_tools_array(builder);
            return; // Success, we're done
        } catch (const common_chat_msg_partial_exception&) {
            // Fall through to try standard DeepSeek patterns
        }
    }

    // If tools array format didn't work, try XML-wrapped format
    builder.move_to(original_pos);
    try {
        parse_deepseek_r1_xml_wrapped(builder);
        return; // Success, we're done
    } catch (const common_chat_msg_partial_exception&) {
        // Fall through to try standard DeepSeek patterns
    }

    // If XML wrapper format didn't work, try standard DeepSeek patterns
    builder.move_to(original_pos);
    try {
        parse_json_tool_calls(
            builder,
            /* block_open= */ tool_calls_begin,
            /* function_regex_start_only= */ std::nullopt,
            function_regex,
            close_regex,
            tool_calls_end);
    } catch (const common_chat_msg_partial_exception&) {
        // If primary regex fails and we're not in partial mode, try fallback regex
        if (!builder.is_partial()) {
            builder.move_to(original_pos);
            try {
                parse_json_tool_calls(
                    builder,
                    /* block_open= */ tool_calls_begin,
                    /* function_regex_start_only= */ std::nullopt,
                    function_regex_no_sep,
                    close_regex,
                    tool_calls_end);
            } catch (const common_chat_msg_partial_exception&) {
                // Try the simple format without markers as final fallback
                builder.move_to(original_pos);
                parse_json_tool_calls(
                    builder,
                    /* block_open= */ std::nullopt,
                    /* function_regex_start_only= */ std::nullopt,
                    function_regex_simple,
                    close_regex_simple,
                    std::nullopt);
            }
        } else {
            throw; // Re-throw for partial mode
        }
    }

    // Add any remaining content (critical for responses without tool calls)
    builder.add_content(builder.consume_rest());
}

// Parse DeepSeek R1 tools array format following original llama.cpp parse_prefixed_json_tool_call_array pattern
static void parse_deepseek_r1_tools_array(common_chat_msg_parser & builder) {
    static const common_regex prefix("function\n```json\n");


    if (auto res = builder.try_find_regex(prefix)) {
        // Parse JSON and manually process tools array to convert arguments to strings
        auto json_result = builder.try_consume_json();
        if (!json_result) {
            throw common_chat_msg_partial_exception("invalid JSON");
        }


        // DeepSeek R1 format has "tools" array, manually process each tool
        if (json_result->json.contains("tools") && json_result->json.at("tools").is_array()) {

            // Manually create tool calls array with string arguments (following original pattern)
            json tools_with_dumped_args = json::array();
            for (const auto& tool : json_result->json.at("tools")) {
                if (tool.contains("name") && tool.contains("arguments")) {
                    json formatted_tool;
                    formatted_tool["name"] = tool.at("name");
                    // Convert arguments object to string (this is what consume_json_with_dumped_args does)
                    formatted_tool["arguments"] = tool.at("arguments").dump();
                    tools_with_dumped_args.push_back(formatted_tool);
                }
            }


            if (!builder.add_tool_calls(tools_with_dumped_args) || !json_result->healing_marker.marker.empty()) {
                throw common_chat_msg_partial_exception("incomplete tool call array");
            }
        } else {
            throw common_chat_msg_partial_exception("tools key not found or not array");
        }

        // Consume closing ```
        builder.try_consume_regex(common_regex("```"));
    } else {
        throw common_chat_msg_partial_exception("function prefix not found");
    }
}

// Parse DeepSeek R1 XML-wrapped format following original Hermes-2-Pro pattern
static void parse_deepseek_r1_xml_wrapped(common_chat_msg_parser & builder) {

    // Pattern for: <tool_call>\nfunction</think>FunctionName\n```json\n{...}\n```\n</tool_call>
    static const common_regex xml_pattern(
        "<tool_call>\\s*"           // Opening XML tag
        "function</think>([^\\n]+)" // Function name after "function</think>"
        "\\s*```json\\s*"           // JSON block start
    );

    if (auto res = builder.try_find_regex(xml_pattern)) {

        // Extract function name from capture group
        std::string function_name = builder.str(res->groups[1]);

        // Parse JSON arguments
        auto json_result = builder.try_consume_json();
        if (!json_result) {
            throw common_chat_msg_partial_exception("invalid JSON in XML wrapper");
        }


        // Create single tool call following original pattern
        json tool_call;
        tool_call["name"] = function_name;
        tool_call["arguments"] = json_result->json.dump();  // Convert to string

        json tool_calls_array = json::array();
        tool_calls_array.push_back(tool_call);


        if (!builder.add_tool_calls(tool_calls_array) || !json_result->healing_marker.marker.empty()) {
            throw common_chat_msg_partial_exception("incomplete XML wrapped tool call");
        }

        // Consume closing ```\n</tool_call>
        builder.try_consume_regex(common_regex("```\\s*</tool_call>"));
    } else {
        throw common_chat_msg_partial_exception("XML wrapper pattern not found");
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

static void common_chat_parse_gpt_oss(common_chat_msg_parser & builder) {
    // TODO @ngxson : this won't work with --special enabled, we should fix that
    builder.try_parse_reasoning("<|channel|>analysis<|message|>", "<|start|>assistant<|channel|>final<|message|>");
    if (!builder.syntax().enable_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
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
        case COMMON_CHAT_FORMAT_GPT_OSS:
            common_chat_parse_gpt_oss(builder);
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
        case COMMON_CHAT_FORMAT_GPT_OSS:      return "GPT-OSS";
        default:                              return "unknown";
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:     return "none";
        case COMMON_REASONING_FORMAT_AUTO:     return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK: return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY: return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

