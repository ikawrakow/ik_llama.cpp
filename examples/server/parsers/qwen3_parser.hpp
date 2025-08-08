#pragma once

#include "json.hpp"
#include "../../common/common.h"
#include <string>
#include <regex>

using json = nlohmann::ordered_json;

//
// Qwen3 Function Calling Parser (XML Hermes format)
// Based on original llama.cpp Hermes 2 Pro parser
//

namespace qwen3 {

// Parse Qwen3 XML-style tool calls: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
static json parse_tool_calls(const std::string& text) {
    json tool_calls = json::array();
    
    try {
        // Look for <tool_call> patterns
        std::regex tool_call_regex(R"(<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>)");
        std::sregex_iterator iter(text.begin(), text.end(), tool_call_regex);
        std::sregex_iterator end;
        
        int call_counter = 0;
        for (; iter != end; ++iter) {
            const std::smatch& match = *iter;
            std::string json_content = match[1].str();
            
            // Clean up the JSON content
            json_content.erase(0, json_content.find_first_not_of(" \t\n\r"));
            json_content.erase(json_content.find_last_not_of(" \t\n\r") + 1);
            
            try {
                // Parse the JSON content
                auto parsed_json = json::parse(json_content);
                
                // Validate required fields
                if (!parsed_json.contains("name") || !parsed_json["name"].is_string()) {
                    continue;
                }
                
                std::string func_name = parsed_json["name"];
                if (func_name.empty()) {
                    continue;
                }
                
                // Extract arguments
                std::string arguments = "{}";
                if (parsed_json.contains("arguments")) {
                    if (parsed_json["arguments"].is_string()) {
                        arguments = parsed_json["arguments"];
                    } else {
                        arguments = parsed_json["arguments"].dump();
                    }
                }
                
                // Generate tool call ID
                std::string tool_id = "qwen3_call_" + std::to_string(++call_counter);
                
                // Create tool call object
                json tool_call = {
                    {"id", tool_id},
                    {"type", "function"},
                    {"function", {
                        {"name", func_name},
                        {"arguments", arguments}
                    }}
                };
                
                tool_calls.push_back(tool_call);
            } catch (const std::exception&) {
                // Skip malformed JSON
                continue;
            }
        }
    } catch (const std::exception&) {
        // Return empty array on any parsing error
        return json::array();
    }
    
    return tool_calls;
}

// Extract clean content by removing tool call tags
static std::string extract_content_during_parsing(const std::string& text, bool is_partial) {
    std::string content = text;
    
    try {
        // Remove <tool_call>...</tool_call> sections
        std::regex tool_call_regex(R"(<tool_call>[\s\S]*?</tool_call>)");
        content = std::regex_replace(content, tool_call_regex, "");
        
        // If partial, check for incomplete tool calls
        if (is_partial) {
            // Look for incomplete <tool_call> without closing tag
            size_t incomplete_pos = content.find("<tool_call>");
            if (incomplete_pos != std::string::npos) {
                // Truncate at the incomplete tool call
                content = content.substr(0, incomplete_pos);
            }
        }
        
        // Only trim leading/trailing whitespace, preserve internal formatting
        content = string_strip(content);
        
    } catch (const std::exception&) {
        // Return original text on regex errors
        return text;
    }
    
    return content;
}

// Legacy cleaning function - kept for compatibility
static std::string clean_content(const std::string& content) {
    return extract_content_during_parsing(content, false);
}

// Helper: Check if content has partial tool call syntax
static bool is_partial_content_advanced(const std::string& content) {
    if (content.empty()) return false;
    
    // Check for incomplete <tool_call> without closing
    size_t open_pos = content.find("<tool_call>");
    if (open_pos != std::string::npos) {
        size_t close_pos = content.find("</tool_call>", open_pos);
        if (close_pos == std::string::npos) {
            return true; // Incomplete tool call
        }
    }
    
    // Check for partial JSON in tool calls
    std::regex incomplete_json_regex(R"(<tool_call>\s*\{[^}]*$)");
    if (std::regex_search(content, incomplete_json_regex)) {
        return true;
    }
    
    return false;
}

} // namespace qwen3