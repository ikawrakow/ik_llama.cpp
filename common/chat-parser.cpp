// Chat parser implementation
#include "chat-parser.h"
#include "../examples/server/parsers/kimi_k2_parser.hpp"
#include "json.hpp"

common_chat_msg_parser::common_chat_msg_parser(const std::string & input, bool is_partial, const common_chat_syntax & syntax)
    : input_(input), is_partial_(is_partial), syntax_(syntax) {
    // Initialize result with default role
    result_.role = "assistant";
}

std::string common_chat_msg_parser::str(const common_string_range & rng) const {
    if (rng.begin > input_.size() || rng.end > input_.size()) {
        throw std::runtime_error("Range out of bounds");
    }
    return input_.substr(rng.begin, rng.end - rng.begin);
}

void common_chat_msg_parser::add_content(const std::string & content) {
    result_.content += content;
}

void common_chat_msg_parser::add_reasoning_content(const std::string & reasoning_content) {
    result_.reasoning_content += reasoning_content;
}

void common_chat_msg_parser::add_tool_call(const common_chat_tool_call & tool_call) {
    result_.tool_calls.push_back(tool_call);
}

void common_chat_msg_parser::clear_tools() {
    result_.tool_calls.clear();
}

std::string common_chat_msg_parser::consume_rest() {
    auto rest = input_.substr(pos_);
    pos_ = input_.size();
    return rest;
}

bool common_chat_msg_parser::try_consume_literal(const std::string & literal) {
    if (pos_ + literal.size() <= input_.size()) {
        if (input_.substr(pos_, literal.size()) == literal) {
            pos_ += literal.size();
            return true;
        }
    }
    return false;
}

bool common_chat_msg_parser::try_parse_reasoning(const std::string & start_think, const std::string & end_think) {
    auto start_pos = input_.find(start_think, pos_);
    if (start_pos == std::string::npos) {
        return false;
    }
    
    auto end_pos = input_.find(end_think, start_pos + start_think.size());
    if (end_pos == std::string::npos) {
        if (is_partial_) {
            // Partial reasoning content
            auto reasoning = input_.substr(start_pos + start_think.size());
            add_reasoning_content(string_strip(reasoning));
            pos_ = input_.size();
            return true;
        }
        return false;
    }
    
    // Extract reasoning content
    auto reasoning = input_.substr(start_pos + start_think.size(), end_pos - start_pos - start_think.size());
    add_reasoning_content(string_strip(reasoning));
    pos_ = end_pos + end_think.size();
    return true;
}

std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_find_literal(const std::string & literal) {
    auto idx = input_.find(literal, pos_);
    if (idx != std::string::npos) {
        find_regex_result res;
        res.prelude = input_.substr(pos_, idx - pos_);
        auto end = idx + literal.size();
        res.groups.emplace_back(common_string_range{idx, end});
        move_to(end);
        return res;
    }
    
    if (is_partial_) {
        idx = string_find_partial_stop(input_, literal);
        if (idx != std::string::npos && idx >= pos_) {
            find_regex_result res;
            res.prelude = input_.substr(pos_, idx - pos_);
            auto end = input_.size();
            res.groups.emplace_back(common_string_range{idx, end});
            move_to(end);
            return res;
        }
    }
    return std::nullopt;
}

void common_chat_msg_parser::parse() {
    switch (syntax_.format) {
        case COMMON_CHAT_FORMAT_KIMI_K2:
            parse_kimi_k2_format();
            break;
        case COMMON_CHAT_FORMAT_GENERIC:
            parse_generic_format();
            break;
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            add_content(consume_rest());
            break;
        default:
            // Fallback to content-only for now
            add_content(consume_rest());
            break;
    }
}

void common_chat_msg_parser::parse_kimi_k2_format() {
    json tool_calls_json = kimi_k2::parse_tool_calls(input_);

    if (is_partial_ && kimi_k2::is_partial_content_advanced(input_)) {
        throw common_chat_msg_partial_exception("partial structured content detected");
    }

    bool has_function_syntax = input_.find("functions.") != std::string::npos;
    bool parsing_succeeded = !tool_calls_json.empty();

    if (has_function_syntax && !parsing_succeeded) {
        throw std::runtime_error("malformed function call syntax detected");
    }

    if (!tool_calls_json.empty()) {
        for (const auto& tc_json : tool_calls_json) {
            try {
                common_chat_tool_call tc;
                tc.id = tc_json.value("id", "");

                if (!tc_json.contains("function") || !tc_json["function"].contains("name")) {
                    continue;
                }

                tc.name = tc_json["function"]["name"];
                if (tc.name.empty()) {
                    continue;
                }

                tc.arguments = tc_json["function"]["arguments"];

                if (!is_partial_ && !tc.arguments.empty()) {
                    try {
                        auto parsed = json::parse(tc.arguments);
                        (void)parsed;
                    } catch (const std::exception&) {
                        continue;
                    }
                }
                add_tool_call(tc);
            } catch (const std::exception&) {
                continue;
            }
        }
        add_content(kimi_k2::clean_content(input_));
    } else {
        add_content(input_);
    }
    pos_ = input_.size();
}

void common_chat_msg_parser::parse_generic_format() {
    add_content(consume_rest());
}

void common_chat_msg_parser::finish() {
    // Any final processing can go here
}

common_chat_msg common_chat_msg_parser::result_and_reset() {
    auto msg = result_;
    result_ = common_chat_msg();
    result_.role = "assistant";
    pos_ = 0;
    return msg;
}

// Main parsing function entry point for original llama.cpp compatibility
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax) {
    common_chat_msg_parser parser(input, is_partial, syntax);
    parser.parse();
    return parser.result();
}

// Content-only parsing for fallback scenarios
void common_chat_parse_content_only(common_chat_msg_parser & builder) {
    builder.add_content(builder.consume_rest());
}