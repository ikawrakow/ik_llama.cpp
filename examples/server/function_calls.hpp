#pragma once

#include "json.hpp"
#include "streaming_chat.hpp"
#include "parsers/kimi_k2_parser.hpp"
#include "../../common/chat.h"
#include "../../common/chat-parser.h"
#include <string>
#include <regex>

using json = nlohmann::ordered_json;

// Function calling interface for Kimi-K2 format
static json parse_kimi_k2_tool_calls(const std::string& text) {
    return kimi_k2::parse_tool_calls(text);
}

static std::string clean_function_calls_from_content(const std::string& content) {
    return kimi_k2::clean_content(content);
}

// New llama.cpp-style content extraction with streaming support
static std::string extract_content_from_mixed_input(const std::string& content, bool is_partial) {
    return kimi_k2::extract_content_during_parsing(content, is_partial);
}

// Incremental parsing for streaming tool calls
static ik_chat_msg parse_chat_message_incremental(const std::string& content, bool is_partial = false) {
    ik_chat_msg msg;
    msg.role = "assistant";
    
    try {
        json tool_calls_json = parse_kimi_k2_tool_calls(content);
        
        // Check for partial content during streaming
        if (is_partial && kimi_k2::is_partial_content_advanced(content)) {
            throw std::runtime_error("partial structured content detected");
        }
        
        // Check for malformed function call syntax
        bool has_function_syntax = content.find("functions.") != std::string::npos;
        bool parsing_succeeded = !tool_calls_json.empty();
        
        if (has_function_syntax && !parsing_succeeded) {
            throw std::runtime_error("malformed function call syntax detected");
        }
        
        // Process successful parsing results  
        if (!tool_calls_json.empty()) {
            for (const auto& tc_json : tool_calls_json) {
                try {
                    ik_chat_tool_call tc;
                    tc.id = tc_json.value("id", "");
                    
                    if (!tc_json.contains("function") || !tc_json["function"].is_object() || !tc_json["function"].contains("name")) {
                        continue;
                    }
                    
                    tc.name = tc_json["function"]["name"];
                    if (tc.name.empty()) {
                        continue;
                    }
                    
                    if (tc_json["function"].contains("arguments")) {
                        tc.arguments = tc_json["function"]["arguments"];
                    } else {
                        tc.arguments = "{}";
                    }
                    
                    // Validate arguments (only if not partial)
                    if (!is_partial && !tc.arguments.empty()) {
                        try {
                            auto parsed = json::parse(tc.arguments);
                            (void)parsed;
                        } catch (const std::exception&) {
                            continue;
                        }
                    }
                    
                    msg.tool_calls.push_back(tc);
                } catch (const std::exception&) {
                    continue;
                }
            }
            
            // Use llama.cpp-style content extraction that handles streaming properly
            msg.content = extract_content_from_mixed_input(content, is_partial);
    } else {
            msg.content = extract_content_from_mixed_input(content, is_partial);
        }
        
    } catch (const std::exception& e) {
        if (!is_partial) {
            // Original llama.cpp fallback pattern - use public API
            common_chat_syntax syntax;
            syntax.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;  // Use content-only format
            
            // Use the public API that handles fallback internally
            common_chat_msg fallback_result = common_chat_parse(content, is_partial, syntax);
            
            // Convert to ik_chat_msg
            msg.tool_calls.clear();
            msg.content = fallback_result.content;
        }
        // If is_partial=true, keep empty result (no content chunks during streaming)
    }
    
    return msg;
}

static std::string generate_tool_call_id() {
    static int counter = 0;
    return "call_" + std::to_string(++counter);
}