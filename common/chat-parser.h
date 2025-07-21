// Chat parser with builder pattern for incremental parsing
#pragma once

#include "chat.h"
#include <optional>
#include <string>
#include <vector>

class common_chat_msg_parser {
    std::string input_;
    bool is_partial_;
    common_chat_syntax syntax_;
    std::string healing_marker_;

    size_t pos_ = 0;
    common_chat_msg result_;

  public:
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
    void clear_tools();
    
    // Parsing utilities
    std::string consume_rest();
    bool try_consume_literal(const std::string & literal);
    bool try_parse_reasoning(const std::string & start_think, const std::string & end_think);
    
    // Main parsing entry point
    void parse();
    
    // Finishing
    void finish();
    
    // Result extraction
    common_chat_msg result_and_reset();

    struct find_regex_result {
        std::string prelude;
        std::vector<common_string_range> groups;
    };

private:
    // Internal parsing helpers
    void parse_kimi_k2_format();
    void parse_generic_format();
    std::optional<find_regex_result> try_find_literal(const std::string & literal);
};

// Content-only parsing for fallback scenarios  
void common_chat_parse_content_only(common_chat_msg_parser & builder);