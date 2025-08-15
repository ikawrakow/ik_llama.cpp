// Chat support with builder pattern for llama.cpp compatibility
#pragma once

#include "common.h"
#include <string>
#include <vector>
#include <functional>

// Forward declarations
struct common_chat_templates;

// Basic data structures compatible with original llama.cpp
struct common_string_range {
    size_t begin;
    size_t end;
    
    common_string_range(size_t begin, size_t end) : begin(begin), end(end) {
        if (begin > end) {
            throw std::runtime_error("Invalid range");
        }
    }
    
    // prevent default ctor
    common_string_range() = delete;
    
    bool empty() const {
        return begin == end;
    }
    
    bool operator==(const common_string_range & other) const {
        return begin == other.begin && end == other.end;
    }
};

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
    
    bool operator!=(const common_chat_tool_call & other) const {
        return !(*this == other);
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

struct common_chat_msg {
    std::string role;
    std::string content;
    std::vector<common_chat_msg_content_part> content_parts = {};
    std::vector<common_chat_tool_call> tool_calls = {};
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && 
               reasoning_content.empty() && tool_name.empty() && tool_call_id.empty();
    }
    
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

    bool operator==(const common_chat_msg & other) const {
        return role == other.role
            && content == other.content
            && content_parts == other.content_parts
            && tool_calls == other.tool_calls
            && reasoning_content == other.reasoning_content
            && tool_name == other.tool_name
            && tool_call_id == other.tool_call_id;
    }
    
    bool operator!=(const common_chat_msg & other) const {
        return !(*this == other);
    }
};

struct common_chat_msg_diff {
    std::string reasoning_content_delta;
    std::string content_delta;
    size_t tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & previous_msg, const common_chat_msg & new_msg);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta
        && tool_call_index == other.tool_call_index
        && tool_call_delta == other.tool_call_delta;
    }
    
    bool operator!=(const common_chat_msg_diff & other) const {
        return !(*this == other);
    }
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    COMMON_CHAT_FORMAT_GENERIC,
    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
    COMMON_CHAT_FORMAT_KIMI_K2,  // Our custom format (keep last for backward compatibility)
};

enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_DEEPSEEK,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
};

struct common_chat_syntax {
    common_chat_format format = COMMON_CHAT_FORMAT_KIMI_K2;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    // Whether reasoning_content should be inlined in the content (e.g. for reasoning_format=deepseek in stream mode)
    bool reasoning_in_content = false;
    bool thinking_forced_open = false;
    bool enable_thinking = false;
    bool enable_tool_calls = true;
};

// Exception for partial parsing
class common_chat_msg_partial_exception : public std::runtime_error {
  public:
    common_chat_msg_partial_exception(const std::string & message) : std::runtime_error(message) {}
};

// Bridge functions to integrate with existing ik_llama.cpp system
// TODO: Uncomment and implement during integration phase
// common_chat_msg ik_to_common_msg(const struct ik_chat_msg & ik_msg);
// struct ik_chat_msg common_to_ik_msg(const common_chat_msg & common_msg);

// Format detection from chat template
common_chat_format common_chat_format_detect(const std::string & chat_template);
const char* common_chat_format_name(common_chat_format format);

// Main parsing function (entry point for original llama.cpp compatibility)
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax);

// Forward declare parser class  
class common_chat_msg_parser;

// Format-specific parsing functions (accessible from chat-parser)
void common_chat_parse_deepseek_r1(common_chat_msg_parser & builder);

