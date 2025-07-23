// Chat parser with builder pattern for incremental parsing
#pragma once

#include "chat.h"
#include "json-partial.h"
#include "regex-partial.h"
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

class common_chat_msg_parser {
    std::string input_;
    bool is_partial_;
    common_chat_syntax syntax_;
    std::string healing_marker_;

    size_t pos_ = 0;
    common_chat_msg result_;

  public:
    struct find_regex_result {
        std::string prelude;
        std::vector<common_string_range> groups;
    };
    
    common_chat_msg_parser(const std::string & input, bool is_partial, const common_chat_syntax & syntax);
    
    // Accessors
    const std::string & input() const { return input_; }
    size_t pos() const { return pos_; }
    const std::string & healing_marker() const { return healing_marker_; }
    const bool & is_partial() const { return is_partial_; }
    const common_chat_msg & result() const { return result_; }
    const common_chat_syntax & syntax() const { return syntax_; }

    // Position manipulation
    void move_to(size_t pos) {
        if (pos > input_.size()) {
            throw std::runtime_error("Invalid position!");
        }
        pos_ = pos;
    }
    
    void move_back(size_t n) {
        if (pos_ < n) {
            throw std::runtime_error("Can't move back that far!");
        }
        pos_ -= n;
    }

    // Get the substring of the input at the given range
    std::string str(const common_string_range & rng) const;

    // Content manipulation
    void add_content(const std::string & content);
    void add_reasoning_content(const std::string & reasoning_content);
    
    // Tool call manipulation
    void add_tool_call(const common_chat_tool_call & tool_call);
    bool add_tool_call(const std::string & name, const std::string & id, const std::string & arguments);
    bool add_tool_call(const json & tool_call);
    bool add_tool_calls(const json & arr);
    void clear_tools();
    
    // Parsing utilities
    std::string consume_rest();
    bool try_consume_literal(const std::string & literal);
    void consume_literal(const std::string & literal);
    bool try_parse_reasoning(const std::string & start_think, const std::string & end_think);
    
    // Regex-based parsing methods (new)
    std::optional<find_regex_result> try_find_regex(const common_regex & regex, size_t from = std::string::npos, bool add_prelude_to_content = true);
    find_regex_result consume_regex(const common_regex & regex);
    std::optional<find_regex_result> try_consume_regex(const common_regex & regex);
    
    // Progressive parsing primitives (for Phase 4)
    std::optional<find_regex_result> try_find_literal(const std::string & literal);
    bool consume_spaces();
    void set_healing_marker(const std::string & marker);
    
    
    // Main parsing entry point
    void parse();
    
    // Finishing
    void finish();
    
    // Result extraction
    common_chat_msg result_and_reset();
    
    // Advanced JSON parsing (following original llama.cpp patterns)
    struct consume_json_result {
        json value;
        bool is_partial;
    };
    
    std::optional<common_json> try_consume_json();
    common_json consume_json();
    consume_json_result consume_json_with_dumped_args(
        const std::vector<std::vector<std::string>>& args_paths = {},
        const std::vector<std::vector<std::string>>& content_paths = {}
    );
    std::optional<consume_json_result> try_consume_json_with_dumped_args(
        const std::vector<std::vector<std::string>>& args_paths = {},
        const std::vector<std::vector<std::string>>& content_paths = {}
    );

private:
    // Internal parsing helpers
    void parse_kimi_k2_format();
    void parse_deepseek_r1_format();
    void parse_generic_format();
    
    // DeepSeek R1 specific tool call parsing
    void parse_deepseek_r1_tool_calls(
        const common_regex & tool_calls_begin,
        const common_regex & function_regex,
        const common_regex & close_regex,
        const common_regex & tool_calls_end);
    
    
    // JSON parsing utilities (enhanced streaming support)
    struct json_parse_result {
        json value;
        bool success;
        bool is_partial;
        std::string healing_marker;
    };
    
    // Partial detection utilities
    bool detect_partial_function_call(const std::string& content);
    void handle_partial_detection();
    
    // Legacy find_literal for compatibility
    std::optional<find_regex_result> try_find_literal_legacy(const std::string & literal);
};

// Main parsing function (public API)
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax);

// Content-only parsing for fallback scenarios (static internal function)