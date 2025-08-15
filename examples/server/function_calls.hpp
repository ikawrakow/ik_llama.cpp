#pragma once

#include "json.hpp"
#include "streaming_chat.hpp"
#include "parsers/kimi_k2_parser.hpp"
#include "parsers/qwen3_parser.hpp"
#include "qwen3_tools.hpp"
#include "deepseek_r1_tools.hpp"
#include "../../common/chat.h"
#include "../../common/chat-parser.h"
#include <string>
#include <regex>

using json = nlohmann::ordered_json;

// Function calling interface for Kimi-K2 format
static json parse_kimi_k2_tool_calls(const std::string& text) {
    return kimi_k2::parse_tool_calls(text);
}

// Function calling interface for Qwen3 format
static json parse_qwen3_tool_calls(const std::string& text) {
    return qwen3::parse_tool_calls(text);
}

static std::string clean_function_calls_from_content(const std::string& content) {
    return kimi_k2::clean_content(content);
}

// New llama.cpp-style content extraction with streaming support
static std::string extract_content_from_mixed_input(const std::string& content, bool is_partial, const std::string& model_name = "") {
    if (is_qwen3_model(model_name)) {
        return qwen3::extract_content_during_parsing(content, is_partial);
    } else if (is_deepseek_r1_model(model_name)) {
        // DeepSeek R1 content extraction - remove <think> tags and tool calls
        std::string result = content;
        
        // Remove <think>...</think> tags
        size_t think_start = 0;
        while ((think_start = result.find("<think>", think_start)) != std::string::npos) {
            size_t think_end = result.find("</think>", think_start);
            if (think_end != std::string::npos) {
                result.erase(think_start, think_end + 8 - think_start);
            } else {
                break;
            }
        }
        
        // Remove DeepSeek R1 tool call syntax
        size_t tool_start = 0;
        while ((tool_start = result.find("<｜tool▁calls▁begin｜>", tool_start)) != std::string::npos) {
            size_t tool_end = result.find("<｜tool▁calls▁end｜>", tool_start);
            if (tool_end != std::string::npos) {
                result.erase(tool_start, tool_end + strlen("<｜tool▁calls▁end｜>") - tool_start);
            } else {
                break;
            }
        }
        
        return result;
    } else {
        return kimi_k2::extract_content_during_parsing(content, is_partial);
    }
}

// Incremental parsing for streaming tool calls with model detection
static ik_chat_msg parse_chat_message_incremental(const std::string& content, bool is_partial = false, const std::string& model_name = "") {
    ik_chat_msg msg;
    msg.role = "assistant";
    
    try {
        json tool_calls_json;
        bool has_function_syntax = false;
        
        // Route parsing based on model type
        if (is_qwen3_model(model_name)) {
            // Use Qwen3 XML parser
            tool_calls_json = parse_qwen3_tool_calls(content);
            
            // Check for partial content during streaming
            if (is_partial && qwen3::is_partial_content_advanced(content)) {
                throw std::runtime_error("partial structured content detected");
            }
            
            // Check for malformed XML tool call syntax
            has_function_syntax = content.find("<tool_call>") != std::string::npos;
        } else if (is_deepseek_r1_model(model_name)) {
            // Use common chat parser for DeepSeek R1
            try {
                common_chat_syntax syntax;
                syntax.format = COMMON_CHAT_FORMAT_DEEPSEEK_R1;
                syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                syntax.reasoning_in_content = true; // Fix for thinking tag termination issue
                syntax.enable_tool_calls = true;
                
                common_chat_msg_parser parser(content, is_partial, syntax);
                parser.parse();
                auto result = parser.result();
                
                // Convert tool calls to JSON format expected by the system
                tool_calls_json = json::array();
                for (const auto& tool_call : result.tool_calls) {
                    json tc;
                    tc["id"] = tool_call.id.empty() ? ("call_" + std::to_string(rand())) : tool_call.id;
                    tc["type"] = "function";
                    tc["function"]["name"] = tool_call.name;
                    tc["function"]["arguments"] = tool_call.arguments;
                    tool_calls_json.push_back(tc);
                }
                
                // Check for malformed DeepSeek R1 tool call syntax
                has_function_syntax = content.find("<｜tool▁calls▁begin｜>") != std::string::npos;
            } catch (const common_chat_msg_partial_exception&) {
                if (is_partial) {
                    throw std::runtime_error("partial structured content detected");
                }
                // If not partial, treat as regular content
                tool_calls_json = json::array();
                has_function_syntax = false;
            }
        } else {
            // Default to Kimi-K2 parser
            tool_calls_json = parse_kimi_k2_tool_calls(content);
            
            // Check for partial content during streaming
            if (is_partial && kimi_k2::is_partial_content_advanced(content)) {
                throw std::runtime_error("partial structured content detected");
            }
            
            // Check for malformed function call syntax
            has_function_syntax = content.find("functions.") != std::string::npos;
        }
        
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
            
            // Use model-specific content extraction
            if (is_qwen3_model(model_name)) {
                msg.content = qwen3::extract_content_during_parsing(content, is_partial);
            } else if (is_deepseek_r1_model(model_name)) {
                msg.content = extract_content_from_mixed_input(content, is_partial, model_name);
            } else {
                msg.content = kimi_k2::extract_content_during_parsing(content, is_partial);
            }
        } else {
            // No tool calls found, extract content
            if (is_qwen3_model(model_name)) {
                msg.content = qwen3::extract_content_during_parsing(content, is_partial);
            } else if (is_deepseek_r1_model(model_name)) {
                msg.content = extract_content_from_mixed_input(content, is_partial, model_name);
            } else {
                msg.content = kimi_k2::extract_content_during_parsing(content, is_partial);
            }
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