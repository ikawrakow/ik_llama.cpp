// Chat parser implementation
#include "chat-parser.h"
#include "../examples/server/parsers/kimi_k2_parser.hpp"
#include "json.hpp"
#include "common.h"

using json = nlohmann::ordered_json;

common_chat_msg_parser::common_chat_msg_parser(const std::string & input, bool is_partial, const common_chat_syntax & syntax)
    : input_(input), is_partial_(is_partial), syntax_(syntax), use_progressive_parsing_(syntax.enable_progressive_parsing) {
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
    if (use_progressive_parsing_) {
        parse_kimi_k2_format_progressive();
        return;
    }
    
    // Legacy parse-then-clean approach
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
    // Pattern: <think>reasoning content</think> followed by regular content
    
    // Try to parse reasoning content first
    if (try_parse_reasoning("<think>", "</think>")) {
        // If reasoning was found, parse remaining content
        add_content(consume_rest());
    } else {
        // No reasoning tags found, treat as regular content
        add_content(consume_rest());
    }
    
    pos_ = input_.size();
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
// Content-only parsing for fallback scenarios (defined in chat.cpp)

// Format detection from chat template patterns (focused on DeepSeek R1 and Kimi K2)
common_chat_format common_chat_format_detect(const std::string & chat_template) {
    if (chat_template.empty()) {
        return COMMON_CHAT_FORMAT_GENERIC;
    }
    
    // Detect DeepSeek R1 format
    if (chat_template.find("<think>") != std::string::npos ||
        chat_template.find("deepseek") != std::string::npos ||
        chat_template.find("DeepSeek") != std::string::npos) {
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

// Progressive Kimi-K2 parser implementation
void common_chat_msg_parser::parse_kimi_k2_format_progressive() {
    // Start with token format detection
    parse_kimi_k2_token_format_progressive();
    
    // Handle any remaining content after progressive parsing
    if (pos_ < input_.length()) {
        add_content(consume_rest());
    }
}

void common_chat_msg_parser::parse_kimi_k2_token_format_progressive() {
    static const std::string begin_marker = "<|tool_calls_section_begin|>";
    static const std::string end_marker = "<|tool_calls_section_end|>";
    
    // Look for tool calls section, add prelude as content
    if (auto result = try_find_literal(begin_marker)) {
        add_content(result->prelude);
        // Parse individual tool calls within section
        static const std::string call_begin = "<|tool_call_begin|>";
        static const std::string call_end = "<|tool_call_end|>";
        static const std::string arg_begin = "<|tool_call_argument_begin|>";
        
        // Parse tool calls within section
        while (pos_ < input_.length()) {
            if (auto call_start = try_find_literal(call_begin)) {
                // Parse single tool call
                auto call_content_start = pos_;
                
                if (auto call_end_result = try_find_literal(call_end)) {
                    // Extract call content
                    std::string call_content = input_.substr(call_content_start, 
                        call_end_result->groups[0].begin - call_content_start);
                    
                    // Parse tool call content
                    size_t arg_start = call_content.find(arg_begin);
                    if (arg_start != std::string::npos) {
                        std::string tool_id_raw = call_content.substr(0, arg_start);
                        std::string arguments_raw = call_content.substr(arg_start + arg_begin.length());
                        
                        // Clean and extract function name
                        std::string tool_id = string_strip(tool_id_raw);
                        std::string arguments = string_strip(arguments_raw);
                        
                        // Extract function name from tool_id (format: functions.{name}:{idx})
                        size_t dot_pos = tool_id.find('.');
                        size_t colon_pos = tool_id.find(':', dot_pos);
                        if (dot_pos != std::string::npos && colon_pos != std::string::npos) {
                            std::string func_name = tool_id.substr(dot_pos + 1, colon_pos - dot_pos - 1);
                            
                            if (!func_name.empty()) {
                                // Validate JSON arguments
                                try {
                                    auto parsed = json::parse(arguments);
                                    
                                    // Create and add tool call
                                    common_chat_tool_call tc;
                                    tc.id = tool_id;
                                    tc.name = func_name;
                                    tc.arguments = arguments;
                                    add_tool_call(tc);
                                } catch (const std::exception&) {
                                    // Invalid JSON, skip this call
                                }
                            }
                        }
                    }
                } else if (is_partial_) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
            } else {
                break; // No more tool calls
            }
        }
        
        // Find end marker
        if (auto end_result = try_find_literal(end_marker)) {
            // Successfully parsed token section
        } else if (is_partial_) {
            set_healing_marker(end_marker);
            throw common_chat_msg_partial_exception("incomplete tool calls section");
        }
    } else {
        // No token format found, try simple format
        parse_kimi_k2_simple_format_progressive();
    }
}

void common_chat_msg_parser::parse_kimi_k2_simple_format_progressive() {
    // Pattern: content functions.name:id{args} content functions.name2:id2{args2} content
    
    while (pos_ < input_.length()) {
        // Look for "functions." pattern, add prelude as content
        if (auto result = try_find_literal("functions.")) {
            add_content(result->prelude);
            // Try to parse complete function call
            if (!try_parse_simple_function_call_progressive()) {
                // Not a valid function call, the literal "functions." was already consumed
                // Continue searching from current position
                continue;
            }
        } else {
            // No more function calls, add remaining content
            add_content(consume_rest());
            break;
        }
    }
}

bool common_chat_msg_parser::try_parse_simple_function_call_progressive() {
    // Parse: name:id{json_args}
    // Current position is right after "functions."
    
    // Extract function name (until ':')
    auto colon_pos = input_.find(':', pos_);
    if (colon_pos == std::string::npos) {
        if (is_partial_) {
            set_healing_marker("functions." + input_.substr(pos_));
            throw common_chat_msg_partial_exception("partial function name");
        }
        return false; // Not a function call
    }
    
    std::string function_name = input_.substr(pos_, colon_pos - pos_);
    if (function_name.empty()) {
        return false;
    }
    
    pos_ = colon_pos + 1;
    
    // Extract ID (until '{')
    auto brace_pos = input_.find('{', pos_);
    if (brace_pos == std::string::npos) {
        if (is_partial_) {
            set_healing_marker("functions." + function_name + ":" + input_.substr(pos_));
            throw common_chat_msg_partial_exception("partial function ID");
        }
        return false;
    }
    
    std::string function_id = input_.substr(pos_, brace_pos - pos_);
    pos_ = brace_pos;
    
    // Parse JSON arguments
    auto json_result = consume_json_args_progressive();
    if (!json_result.success) {
        if (is_partial_ && json_result.is_partial) {
            throw common_chat_msg_partial_exception("partial JSON arguments");
        }
        return false;
    }
    
    // Create complete tool call ID
    std::string tool_id = "functions." + function_name + ":" + function_id;
    
    // Add successful tool call
    common_chat_tool_call tc;
    tc.id = tool_id;
    tc.name = function_name;
    tc.arguments = json_result.value.dump();
    add_tool_call(tc);
    
    return true;
}

common_chat_msg_parser::json_parse_result common_chat_msg_parser::consume_json_args_progressive() {
    size_t start_pos = pos_;
    
    if (pos_ >= input_.length() || input_[pos_] != '{') {
        return {json(), false, is_partial_, ""};
    }
    
    // Find matching closing brace
    int brace_count = 0;
    size_t json_end = pos_;
    bool in_string = false;
    bool escaped = false;
    
    while (json_end < input_.length()) {
        char c = input_[json_end];
        
        if (!escaped && c == '"' && !in_string) {
            in_string = true;
        } else if (!escaped && c == '"' && in_string) {
            in_string = false;
        } else if (!in_string) {
            if (c == '{') brace_count++;
            else if (c == '}') brace_count--;
        }
        
        escaped = (!escaped && c == '\\');
        json_end++;
        
        if (brace_count == 0) break;
    }
    
    if (brace_count > 0) {
        // Incomplete JSON
        if (is_partial_) {
            std::string partial_json = input_.substr(start_pos, json_end - start_pos);
            return {json(), false, true, partial_json};
        }
        return {json(), false, false, ""};
    }
    
    // Extract and parse JSON
    std::string json_str = input_.substr(start_pos, json_end - start_pos);
    pos_ = json_end;
    
    try {
        json parsed = json::parse(json_str);
        return {parsed, true, false, ""};
    } catch (const std::exception&) {
        return {json(), false, false, ""};
    }
}

void common_chat_msg_parser::parse_kimi_k2_xml_format_progressive() {
    // This would implement XML parsing - for now, fall back to simple format
    parse_kimi_k2_simple_format_progressive();
}

void common_chat_msg_parser::parse_xml_tool_call_progressive() {
    // XML parsing implementation would go here
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