// Chat parser implementation
#include "chat-parser.h"
#include "../examples/server/parsers/kimi_k2_parser.hpp"
#include "json.hpp"
#include "common.h"

using json = nlohmann::ordered_json;

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

bool common_chat_msg_parser::add_tool_call(const std::string & name, const std::string & id, const std::string & arguments) {
    if (name.empty()) {
        return false;
    }
    
    common_chat_tool_call tool_call;
    tool_call.name = name;
    tool_call.arguments = arguments;
    tool_call.id = id;
    
    result_.tool_calls.emplace_back(tool_call);
    return true;
}

bool common_chat_msg_parser::add_tool_call(const json & tool_call) {
    std::string name = tool_call.contains("name") ? tool_call.at("name") : "";
    std::string id = tool_call.contains("id") ? tool_call.at("id") : "";
    std::string arguments = tool_call.contains("arguments") ? tool_call.at("arguments") : "";
    return add_tool_call(name, id, arguments);
}

bool common_chat_msg_parser::add_tool_calls(const json & arr) {
    for (const auto & item : arr) {
        if (!add_tool_call(item)) {
            return false;
        }
    }
    return true;
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

std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_find_literal_legacy(const std::string & literal) {
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
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:
            parse_deepseek_r1_format();
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

void common_chat_msg_parser::parse_deepseek_r1_format() {
    // DeepSeek R1 format supports <think> tags for reasoning content
    try_parse_reasoning("<think>", "</think>");
    
    if (!syntax_.enable_tool_calls) {
        add_content(consume_rest());
        return;
    }

    // DeepSeek R1 tool call patterns from original llama.cpp
    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?function<｜tool▁sep｜>([^\n]+)\n```json\n");
    static const common_regex close_regex("```[\\s\\r\\n]*<｜tool▁call▁end｜>");

    parse_deepseek_r1_tool_calls(tool_calls_begin, function_regex, close_regex, tool_calls_end);
}

void common_chat_msg_parser::parse_deepseek_r1_tool_calls(
    const common_regex & tool_calls_begin,
    const common_regex & function_regex,
    const common_regex & close_regex,
    const common_regex & tool_calls_end) {
    
    // Helper function to wrap code as JSON arguments (ported from original llama.cpp)
    auto wrap_code_as_arguments = [this](const std::string & code) -> std::string {
        std::string arguments;
        if (is_partial_) {
            arguments = (json {{"code", code + healing_marker_}}).dump();
            auto idx = arguments.find(healing_marker_);
            if (idx != std::string::npos) {
                arguments.resize(idx);
            }
        } else {
            arguments = (json {{"code", code}}).dump();
        }
        return arguments;
    };

    auto parse_tool_calls = [&]() {
        size_t from = std::string::npos;
        while (true) {
            auto res = try_find_regex(function_regex, from);
            if (res) {
                // Extract function name from regex group 1
                std::string name = str(res->groups[1]);
                from = std::string::npos;
                
                if (name.empty()) {
                    from = res->groups[0].begin + 1;
                    continue;
                }

                auto maybe_raw_python = name == "python";
                if (input_[pos_] == '{' || !maybe_raw_python) {
                    if (auto arguments = try_consume_json_with_dumped_args({{}})) {
                        if (!add_tool_call(name, "", arguments->value) || arguments->is_partial) {
                            throw common_chat_msg_partial_exception("incomplete tool call");
                        }
                        try_consume_regex(close_regex);
                    }
                    continue;
                }
                if (maybe_raw_python) {
                    auto arguments = wrap_code_as_arguments(consume_rest());
                    if (!add_tool_call(name, "", arguments)) {
                        throw common_chat_msg_partial_exception("incomplete tool call");
                    }
                    return;
                }
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
            break;
        }
        try_consume_regex(tool_calls_end);
        consume_spaces();
        add_content(consume_rest());
    };
    
    if (auto res = try_find_regex(tool_calls_begin)) {
        parse_tool_calls();
    } else {
        add_content(consume_rest());
    }
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

// Content-only parsing for fallback scenarios

// Format detection from chat template patterns (focused on DeepSeek R1 and Kimi K2)
common_chat_format common_chat_format_detect(const std::string & chat_template) {
    if (chat_template.empty()) {
        return COMMON_CHAT_FORMAT_GENERIC;
    }
    
    // Detect DeepSeek R1 format (following original llama.cpp detection logic)
    if (chat_template.find("<｜tool▁calls▁begin｜>") != std::string::npos) {
        return COMMON_CHAT_FORMAT_DEEPSEEK_R1;
    }
    
    // Detect Kimi K2 format (our custom format)
    if (chat_template.find("kimi") != std::string::npos ||
        chat_template.find("Kimi") != std::string::npos ||
        chat_template.find("functions.") != std::string::npos) {
        return COMMON_CHAT_FORMAT_KIMI_K2;
    }
    
    // Default to generic format for unknown templates
    return COMMON_CHAT_FORMAT_GENERIC;
}

// Progressive parsing primitive - find literal (following original llama.cpp pattern)
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

bool common_chat_msg_parser::consume_spaces() {
    bool consumed = false;
    while (pos_ < input_.length() && std::isspace(input_[pos_])) {
        pos_++;
        consumed = true;
    }
    return consumed;
}

void common_chat_msg_parser::set_healing_marker(const std::string & marker) {
    healing_marker_ = marker;
}


// Enhanced JSON parsing methods (following original llama.cpp patterns exactly)
std::optional<common_json> common_chat_msg_parser::try_consume_json() {
    auto it = input_.cbegin() + pos_;
    const auto end = input_.cend();
    common_json result;
    if (!common_json_parse(it, end, healing_marker_, result)) {
        return std::nullopt;
    }
    pos_ = std::distance(input_.cbegin(), it);
    if (result.healing_marker.marker.empty()) {
        // No healing marker, just return the parsed json
        return result;
    }
    if (!is_partial()) {
        throw common_chat_msg_partial_exception("JSON");
    }
    return result;
}

common_json common_chat_msg_parser::consume_json() {
    if (auto result = try_consume_json()) {
        return *result;
    }
    throw common_chat_msg_partial_exception("JSON");
}

common_chat_msg_parser::consume_json_result common_chat_msg_parser::consume_json_with_dumped_args(
    const std::vector<std::vector<std::string>>& args_paths,
    const std::vector<std::vector<std::string>>& content_paths
) {
    if (auto result = try_consume_json_with_dumped_args(args_paths, content_paths)) {
        return *result;
    }
    throw common_chat_msg_partial_exception("JSON");
}

std::optional<common_chat_msg_parser::consume_json_result> common_chat_msg_parser::try_consume_json_with_dumped_args(
    const std::vector<std::vector<std::string>>& args_paths,
    const std::vector<std::vector<std::string>>& content_paths
) {
    auto partial = try_consume_json();
    if (!partial) {
        return std::nullopt;
    }
    auto is_arguments_path = [&](const std::vector<std::string> & path) {
        return std::find(args_paths.begin(), args_paths.end(), path) != args_paths.end();
    };
    auto is_content_path = [&](const std::vector<std::string> & path) {
        return std::find(content_paths.begin(), content_paths.end(), path) != content_paths.end();
    };

    if (partial->healing_marker.marker.empty()) {
        if (args_paths.empty()) {
            // No arguments to dump, and JSON was parsed fully.
            return consume_json_result {
                partial->json,
                /* .is_partial = */ false,
            };
        }
        if (is_arguments_path({})) {
            // Entire JSON is the arguments and was parsed fully.
            return consume_json_result {
                partial->json.dump(),
                /* .is_partial = */ false,
            };
        }
        // TODO: Implement full path-based argument dumping logic from original
        // For now, return the parsed JSON as-is
        return consume_json_result {
            partial->json,
            /* .is_partial = */ false,
        };
    }
    
    // Has healing marker - this is partial JSON
    // TODO: Implement sophisticated partial JSON handling with path-based dumping
    // For now, return partial result
    return consume_json_result {
        partial->json,
        /* .is_partial = */ true,
    };
}

bool common_chat_msg_parser::detect_partial_function_call(const std::string& content) {
    if (content.empty()) return false;
    
    // Enhanced partial detection patterns
    static const std::vector<std::string> partial_patterns = {
        "functions",
        "functions.",
        "<tool_call",
        "<tool_call>",
        "<invoke",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>"
    };
    
    for (const auto& pattern : partial_patterns) {
        if (content.substr(0, pattern.length()) == pattern && content.length() <= pattern.length() + 50) {
            return true;
        }
    }
    
    return false;
}

void common_chat_msg_parser::handle_partial_detection() {
    if (!is_partial_) return;
    
    // Check for various partial patterns
    std::string remaining = input_.substr(pos_);
    
    if (remaining.empty()) return;
    
    // Detect partial function calls
    if (detect_partial_function_call(remaining)) {
        set_healing_marker(remaining);
        throw common_chat_msg_partial_exception("partial function call detected");
    }
    
    // Enhanced partial JSON detection
    if (remaining.find('{') != std::string::npos) {
        size_t brace_pos = remaining.find('{');
        std::string json_part = remaining.substr(brace_pos);
        
        // Check if JSON is incomplete
        int brace_count = 0;
        bool in_string = false;
        bool escaped = false;
        bool is_incomplete = true;
        
        for (size_t i = 0; i < json_part.length(); i++) {
            char c = json_part[i];
            
            if (!escaped) {
                if (c == '"' && !in_string) {
                    in_string = true;
                } else if (c == '"' && in_string) {
                    in_string = false;
                } else if (!in_string) {
                    if (c == '{') brace_count++;
                    else if (c == '}') brace_count--;
                }
            }
            
            escaped = (!escaped && c == '\\');
            
            if (brace_count == 0) {
                is_incomplete = false;
                break;
            }
        }
        
        if (is_incomplete) {
            set_healing_marker(json_part);
            throw common_chat_msg_partial_exception("partial JSON detected");
        }
    }
}

// Regex-based parsing methods (ported from original llama.cpp)
std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_find_regex(const common_regex & regex, size_t from, bool add_prelude_to_content) {
    auto m = regex.search(input_, from == std::string::npos ? pos_ : from);
    if (m.type == COMMON_REGEX_MATCH_TYPE_NONE) {
        return std::nullopt;
    }
    auto prelude = input_.substr(pos_, m.groups[0].begin - pos_);
    pos_ = m.groups[0].end;

    if (add_prelude_to_content) {
        add_content(prelude);
    }
    if (m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) {
        if (is_partial()) {
            throw common_chat_msg_partial_exception(regex.str());
        }
        return std::nullopt;
    }
    return find_regex_result{prelude, m.groups};
}

common_chat_msg_parser::find_regex_result common_chat_msg_parser::consume_regex(const common_regex & regex) {
    auto result = try_find_regex(regex);
    if (!result) {
        throw std::runtime_error("Expected regex not found: " + regex.str());
    }
    return *result;
}

std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_consume_regex(const common_regex & regex) {
    return try_find_regex(regex, pos_, false);
}

void common_chat_msg_parser::consume_literal(const std::string & literal) {
    if (!try_consume_literal(literal)) {
        throw std::runtime_error("Expected literal not found: " + literal);
    }
}

// Get format name for debugging/logging (implemented in chat.cpp)