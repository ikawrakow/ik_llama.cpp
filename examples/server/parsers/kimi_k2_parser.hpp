#pragma once

#include "json.hpp"
#include <string>
#include <regex>

using json = nlohmann::ordered_json;

//
// Kimi-K2 Function Calling Parser
// Handles both native token format and simple format
//

namespace kimi_k2 {

// Constants for token format markers
static constexpr const char* TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>";
static constexpr const char* TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>";
static constexpr const char* TOOL_CALL_BEGIN = "<|tool_call_begin|>";
static constexpr const char* TOOL_CALL_END = "<|tool_call_end|>";
static constexpr const char* TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>";

// Constants for XML format markers
static constexpr const char* XML_TOOL_CALL_OPEN = "<tool_call>";
static constexpr const char* XML_TOOL_CALL_CLOSE = "</tool_call>";
static constexpr const char* XML_INVOKE_OPEN_PREFIX = "<invoke name=\"";
static constexpr const char* XML_INVOKE_CLOSE = "</invoke>";
static constexpr const char* XML_PARAMETER_OPEN_PREFIX = "<parameter name=\"";
static constexpr const char* XML_PARAMETER_CLOSE = "</parameter>";

// Constants for simple format patterns
static constexpr const char* FUNCTIONS_PREFIX = "functions.";

// Helper functions to get marker lengths at compile time
static constexpr size_t get_marker_length(const char* marker) {
    size_t len = 0;
    while (marker[len] != '\0') ++len;
    return len;
}

static constexpr size_t TOOL_CALLS_SECTION_BEGIN_LEN = get_marker_length(TOOL_CALLS_SECTION_BEGIN);
static constexpr size_t TOOL_CALLS_SECTION_END_LEN = get_marker_length(TOOL_CALLS_SECTION_END);
static constexpr size_t TOOL_CALL_BEGIN_LEN = get_marker_length(TOOL_CALL_BEGIN);
static constexpr size_t TOOL_CALL_END_LEN = get_marker_length(TOOL_CALL_END);
static constexpr size_t TOOL_CALL_ARGUMENT_BEGIN_LEN = get_marker_length(TOOL_CALL_ARGUMENT_BEGIN);
static constexpr size_t XML_TOOL_CALL_OPEN_LEN = get_marker_length(XML_TOOL_CALL_OPEN);
static constexpr size_t XML_TOOL_CALL_CLOSE_LEN = get_marker_length(XML_TOOL_CALL_CLOSE);
static constexpr size_t XML_PARAMETER_CLOSE_LEN = get_marker_length(XML_PARAMETER_CLOSE);
static constexpr size_t FUNCTIONS_PREFIX_LEN = get_marker_length(FUNCTIONS_PREFIX);

// Helper function to trim whitespace and quotes
static std::string trim_and_unquote(const std::string& str) {
    std::string result = str;
    
    // Trim whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    
    // Remove surrounding quotes if present
    if (result.length() >= 2 && result.front() == '"' && result.back() == '"') {
        result = result.substr(1, result.length() - 2);
    }
    
    return result;
}

// Parse Kimi-K2 native token format (format: <|tool_calls_section_begin|>...<|tool_calls_section_end|>)
static json parse_token_function_calls(const std::string& text) {
    json tool_calls = json::array();
    
    try {
        // Look for tool calls section
        size_t section_start = text.find(TOOL_CALLS_SECTION_BEGIN);
        if (section_start == std::string::npos) {
            return tool_calls;
        }
        
        size_t section_end = text.find(TOOL_CALLS_SECTION_END, section_start);
        if (section_end == std::string::npos) {
            return tool_calls;
        }
        
        // Extract section content
        std::string section = text.substr(section_start + TOOL_CALLS_SECTION_BEGIN_LEN, 
                                        section_end - section_start - TOOL_CALLS_SECTION_BEGIN_LEN);
        
        // Parse individual tool calls
        size_t pos = 0;
        while (pos < section.length()) {
            size_t call_start = section.find(TOOL_CALL_BEGIN, pos);
            if (call_start == std::string::npos) break;
            
            size_t call_end = section.find(TOOL_CALL_END, call_start);
            if (call_end == std::string::npos) break;
            
            std::string call_content = section.substr(call_start + TOOL_CALL_BEGIN_LEN, 
                                                    call_end - call_start - TOOL_CALL_BEGIN_LEN);
            
            // Parse tool call content
            size_t arg_start = call_content.find(TOOL_CALL_ARGUMENT_BEGIN);
            if (arg_start != std::string::npos) {
                std::string tool_id_raw = call_content.substr(0, arg_start);
                std::string arguments_raw = call_content.substr(arg_start + TOOL_CALL_ARGUMENT_BEGIN_LEN);
                
                // Clean tool_id and arguments
                std::string tool_id = tool_id_raw;
                std::string arguments = arguments_raw;
                
                // Trim whitespace but preserve the ID format
                tool_id.erase(0, tool_id.find_first_not_of(" \t\n\r"));
                tool_id.erase(tool_id.find_last_not_of(" \t\n\r") + 1);
                arguments.erase(0, arguments.find_first_not_of(" \t\n\r"));
                arguments.erase(arguments.find_last_not_of(" \t\n\r") + 1);
                
                // Extract function name from tool_id (format: functions.{name}:{idx})
                std::string func_name = "";
                size_t dot_pos = tool_id.find('.');
                size_t colon_pos = tool_id.find(':', dot_pos);
                if (dot_pos != std::string::npos && colon_pos != std::string::npos) {
                    func_name = tool_id.substr(dot_pos + 1, colon_pos - dot_pos - 1);
                }
                
                // Skip if function name is empty
                if (func_name.empty()) {
                    pos = call_end + TOOL_CALL_END_LEN;
                    continue;
                }
                
                // Validate arguments is valid JSON
                try {
                    auto parsed = json::parse(arguments);
                    (void)parsed; // Suppress unused variable warning
                } catch (const std::exception&) {
                    pos = call_end + TOOL_CALL_END_LEN;
                    continue;
                }
                
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
            }
            
            pos = call_end + TOOL_CALL_END_LEN;
        }
    } catch (const std::exception&) {
        // Return empty array on any parsing error
        return json::array();
    }
    
    return tool_calls;
}

// Parse XML-style function calls: <tool_call><invoke name="..."><parameter name="..." >...</parameter></invoke></tool_call>
static json parse_xml_function_calls(const std::string& text) {
    json tool_calls = json::array();
    
    try {
        size_t pos = 0;
        while ((pos = text.find(XML_TOOL_CALL_OPEN, pos)) != std::string::npos) {
            size_t tool_call_start = pos;
            size_t tool_call_end = text.find(XML_TOOL_CALL_CLOSE, tool_call_start);
            if (tool_call_end == std::string::npos) {
                pos = tool_call_start + XML_TOOL_CALL_OPEN_LEN;
                continue;
            }
            
            std::string tool_call_content = text.substr(tool_call_start + XML_TOOL_CALL_OPEN_LEN, 
                                                      tool_call_end - tool_call_start - XML_TOOL_CALL_OPEN_LEN);
            
            // Look for <invoke name="function_name">
            size_t invoke_start = tool_call_content.find(XML_INVOKE_OPEN_PREFIX);
            if (invoke_start == std::string::npos) {
                pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
                continue;
            }
            
            // Find the opening quote after "name="
            size_t quote_start = tool_call_content.find("\"", invoke_start);
            if (quote_start == std::string::npos) {
                pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
                continue;
            }
            
            // Find the closing quote
            size_t quote_end = tool_call_content.find("\"", quote_start + 1);
            if (quote_end == std::string::npos) {
                pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
                continue;
            }
            
            // Extract function name between quotes
            std::string func_name = tool_call_content.substr(quote_start + 1, quote_end - quote_start - 1);
            if (func_name.empty()) {
                pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
                continue;
            }
            
            // Look for closing >
            size_t invoke_close = tool_call_content.find(">", quote_end);
            if (invoke_close == std::string::npos) {
                pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
                continue;
            }
            
            // Find </invoke>
            size_t invoke_end = tool_call_content.find(XML_INVOKE_CLOSE);
            if (invoke_end == std::string::npos) {
                pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
                continue;
            }
            
            // Extract parameters
            std::string params_section = tool_call_content.substr(invoke_close + 1, invoke_end - invoke_close - 1);
            
            // Parse parameters and build JSON arguments
            json args = json::object();
            size_t param_pos = 0;
            while ((param_pos = params_section.find(XML_PARAMETER_OPEN_PREFIX, param_pos)) != std::string::npos) {
                // Find the opening quote after "name="
                size_t param_quote_start = params_section.find("\"", param_pos);
                if (param_quote_start == std::string::npos) break;
                
                // Find the closing quote
                size_t param_quote_end = params_section.find("\"", param_quote_start + 1);
                if (param_quote_end == std::string::npos) break;
                
                std::string param_name = params_section.substr(param_quote_start + 1, param_quote_end - param_quote_start - 1);
                
                size_t param_content_start = params_section.find(">", param_quote_end);
                if (param_content_start == std::string::npos) break;
                param_content_start++;
                
                size_t param_content_end = params_section.find(XML_PARAMETER_CLOSE, param_content_start);
                if (param_content_end == std::string::npos) break;
                
                std::string param_value = params_section.substr(param_content_start, param_content_end - param_content_start);
                
                // Clean up parameter value (trim whitespace)
                param_value.erase(0, param_value.find_first_not_of(" \t\n\r"));
                param_value.erase(param_value.find_last_not_of(" \t\n\r") + 1);
                
                args[param_name] = param_value;
                param_pos = param_content_end + XML_PARAMETER_CLOSE_LEN;
            }
            
            // Generate tool call ID
            static int xml_call_counter = 0;
            std::string tool_id = "call_xml_" + std::to_string(++xml_call_counter);
            
            // Create tool call object
            json tool_call = {
                {"id", tool_id},
                {"type", "function"},
                {"function", {
                    {"name", func_name},
                    {"arguments", args.dump()}
                }}
            };
            
            tool_calls.push_back(tool_call);
            pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
        }
    } catch (const std::exception&) {
        // Return empty array on any parsing error
        return json::array();
    }
    
    return tool_calls;
}

// Parse simple function call format: functions.function_name:index{json_args}
static json parse_simple_function_calls(const std::string& text) {
    json tool_calls = json::array();
    
    try {
        // Look for patterns like "functions.function_name:index{json_args}"
        size_t pos = 0;
        
        while ((pos = text.find(FUNCTIONS_PREFIX, pos)) != std::string::npos) {
            size_t func_start = pos + FUNCTIONS_PREFIX_LEN;
            
            // Find the colon that separates function name from index
            size_t colon_pos = text.find(':', func_start);
            if (colon_pos == std::string::npos) {
                pos = func_start;
                continue;
            }
            
            // Extract function name
            std::string func_name = text.substr(func_start, colon_pos - func_start);
            
            // Skip if function name is empty
            if (func_name.empty()) {
                pos = colon_pos;
                continue;
            }
            
            // Extract index
            size_t index_start = colon_pos + 1;
            size_t brace_pos = text.find('{', index_start);
            if (brace_pos == std::string::npos) {
                pos = colon_pos;
                continue;
            }
            
            std::string index_str = text.substr(index_start, brace_pos - index_start);
            
            // Find the matching closing brace
            int brace_count = 1;
            size_t end_pos = brace_pos + 1;
            while (end_pos < text.length() && brace_count > 0) {
                if (text[end_pos] == '{') brace_count++;
                else if (text[end_pos] == '}') brace_count--;
                end_pos++;
            }
            
            if (brace_count == 0) {
                // Extract arguments JSON
                std::string args_json = text.substr(brace_pos, end_pos - brace_pos);
                
                // Validate arguments is valid JSON
                try {
                    auto parsed = json::parse(args_json);
                    (void)parsed; // Suppress unused variable warning
                } catch (const std::exception&) {
                    pos = end_pos;
                    continue;
                }
                
                // Generate tool call ID with actual index from the call
                std::string tool_id = "functions." + func_name + ":" + index_str;
                
                // Create tool call object
                json tool_call = {
                    {"id", tool_id},
                    {"type", "function"},
                    {"function", {
                        {"name", func_name},
                        {"arguments", args_json}
                    }}
                };
                
                tool_calls.push_back(tool_call);
            }
            
            pos = end_pos;
        }
    } catch (const std::exception&) {
        // Return empty array on any parsing error
        return json::array();
    }
    
    return tool_calls;
}

// Main function to parse Kimi-K2 native tool calls
static json parse_tool_calls(const std::string& text) {
    try {
        // Check if we have token format markers
        bool has_token_start = text.find(TOOL_CALLS_SECTION_BEGIN) != std::string::npos;
        bool has_token_end = text.find(TOOL_CALLS_SECTION_END) != std::string::npos;
        bool has_token_section = has_token_start && has_token_end;
        
        json result = json::array();
        
        // If we have a token start but no end, it's malformed - return empty
        if (has_token_start && !has_token_end) {
            return result;
        }
        
        if (has_token_section) {
            // Parse token format
            json token_calls = parse_token_function_calls(text);
            
            // For mixed format, also check for simple calls outside the token section
            std::string content_for_simple = text;
            size_t section_start = content_for_simple.find(TOOL_CALLS_SECTION_BEGIN);
            size_t section_end = content_for_simple.find(TOOL_CALLS_SECTION_END);
            if (section_start != std::string::npos && section_end != std::string::npos) {
                // Remove the token section to avoid double-parsing
                content_for_simple = content_for_simple.substr(0, section_start) + 
                                   content_for_simple.substr(section_end + TOOL_CALLS_SECTION_END_LEN);
            }
            
            json simple_calls = parse_simple_function_calls(content_for_simple);
            
            // Combine results
            result = token_calls;
            for (const auto& call : simple_calls) {
                result.push_back(call);
            }
        } else {
            // No token format, try both XML and simple formats
            json xml_calls = parse_xml_function_calls(text);
            json simple_calls = parse_simple_function_calls(text);
            
            // Combine results (XML takes precedence if both exist)
            result = xml_calls;
            for (const auto& call : simple_calls) {
                result.push_back(call);
            }
        }
        
        return result;
    } catch (const std::exception&) {
        // Return empty array on any error
        return json::array();
    }
}

// llama.cpp-style content extraction: separate content during parsing
static std::string extract_content_during_parsing(const std::string& text, bool is_partial) {
    std::string content;
    size_t last_content_end = 0;
    
    // Process XML-style tool calls first: <tool_call>...</tool_call>
    size_t xml_pos = 0;
    while ((xml_pos = text.find(XML_TOOL_CALL_OPEN, xml_pos)) != std::string::npos) {
        // Add content before this tool call
        content += text.substr(last_content_end, xml_pos - last_content_end);
        
        // Skip to end of tool call
        size_t tool_call_end = text.find(XML_TOOL_CALL_CLOSE, xml_pos);
        if (tool_call_end != std::string::npos) {
            xml_pos = tool_call_end + XML_TOOL_CALL_CLOSE_LEN;
            last_content_end = xml_pos;
        } else {
            // Incomplete tool call - stop here if partial
            if (is_partial) {
                return string_strip(content);
            }
            xml_pos += XML_TOOL_CALL_OPEN_LEN;
        }
    }
    
    // Process token format sections first: <|tool_calls_section_begin|>...<|tool_calls_section_end|>
    size_t section_start = text.find(TOOL_CALLS_SECTION_BEGIN, last_content_end);
    if (section_start != std::string::npos) {
        // Add content before section
        content += text.substr(last_content_end, section_start - last_content_end);
        
        size_t section_end = text.find(TOOL_CALLS_SECTION_END, section_start);
        if (section_end != std::string::npos) {
            // Skip entire section
            last_content_end = section_end + TOOL_CALLS_SECTION_END_LEN;
        } else if (is_partial) {
            // Incomplete section during streaming - stop here
            return string_strip(content);
        }
    }
    
    // Process simple function calls: functions.name:id{json}
    size_t func_pos = last_content_end;
    while ((func_pos = text.find(FUNCTIONS_PREFIX, func_pos)) != std::string::npos) {
        // Add content before this function call
        content += text.substr(last_content_end, func_pos - last_content_end);
        
        // Find the opening brace for arguments
        size_t brace_pos = text.find('{', func_pos);
        if (brace_pos == std::string::npos) {
            // No opening brace found
            if (is_partial) {
                // This might be incomplete function call - stop here
                return string_strip(content);
            }
            func_pos += FUNCTIONS_PREFIX_LEN;
            continue;
        }
        
        // Find matching closing brace
        int brace_count = 1;
        size_t end_pos = brace_pos + 1;
        while (end_pos < text.length() && brace_count > 0) {
            if (text[end_pos] == '{') brace_count++;
            else if (text[end_pos] == '}') brace_count--;
            end_pos++;
        }
        
        if (brace_count == 0) {
            // Complete function call - skip it
            func_pos = end_pos;
            last_content_end = func_pos;
        } else {
            // Incomplete function call
            if (is_partial) {
                // During streaming, stop at incomplete function call
                return string_strip(content);
            }
            // Not streaming, skip partial pattern
            func_pos = brace_pos + 1;
        }
    }
    
    // Add any remaining content after all tool calls
    if (last_content_end < text.length()) {
        content += text.substr(last_content_end);
    }
    
    return string_strip(content);
}

// Legacy cleaning function - kept for compatibility
static std::string clean_content(const std::string& content) {
    // Use the new extraction method with is_partial=false for backward compatibility
    return extract_content_during_parsing(content, false);
}

// Helper: Find matching closing brace 
static size_t find_matching_brace(const std::string& content, size_t start_pos) {
    if (start_pos >= content.length() || content[start_pos] != '{') {
        return std::string::npos;
    }
    
    int brace_count = 1;
    bool in_string = false;
    bool escaped = false;
    
    for (size_t i = start_pos + 1; i < content.length() && brace_count > 0; i++) {
        char c = content[i];
        
        if (!in_string) {
            if (c == '{') brace_count++;
            else if (c == '}') brace_count--;
            else if (c == '"') in_string = true;
        } else {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        }
        
        if (brace_count == 0) return i;
    }
    
    return std::string::npos;
}

// Helper: Check if JSON starting at position is incomplete (like original healing detection)
static bool is_incomplete_json(const std::string& json_str) {
    if (json_str.empty() || json_str[0] != '{') return true;
    
    try {
        // Try to parse as-is first
        auto parsed = json::parse(json_str);
        return false; // Complete JSON
    } catch (const std::exception&) {
        // Failed to parse - likely incomplete
        
        // Check for common incomplete patterns
        std::string trimmed = json_str;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
        trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
        
        // Incomplete patterns that should be detected as partial
        if (trimmed == "{") return true;
        if (trimmed.back() == ':') return true;
        if (trimmed.back() == ',') return true;
        if (trimmed.back() == '"' && trimmed.find('"', 1) == trimmed.length() - 1) return true;
        
        // Count braces to detect imbalance
        int brace_count = 0;
        bool in_string = false;
        bool escaped = false;
        
        for (char c : trimmed) {
            if (!in_string) {
                if (c == '{') brace_count++;
                else if (c == '}') brace_count--;
                else if (c == '"') in_string = true;
            } else {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    in_string = false;
                }
            }
        }
        
        return brace_count > 0 || in_string; // Unbalanced or incomplete string
    }
}

// Helper: Check if JSON starting at specific position is complete
static bool is_json_complete_from_position(const std::string& content, size_t start_pos) {
    if (start_pos >= content.length() || content[start_pos] != '{') return false;
    
    size_t end_pos = find_matching_brace(content, start_pos);
    if (end_pos == std::string::npos) return false;
    
    std::string json_part = content.substr(start_pos, end_pos - start_pos + 1);
    return !is_incomplete_json(json_part);
}

// Enhanced partial detection based on original llama.cpp patterns
// Detects various streaming edge cases that indicate incomplete content
static bool is_partial_content_advanced(const std::string& content) {
    if (content.empty()) return false;
    
    // 1. Basic function syntax partials (like original llama.cpp partial JSON detection)
    if (content == "functions" || content == "func") {
        return true;
    }
    
    // Check if content ends with incomplete function syntax (anywhere in content)
    if (content.find("functions") != std::string::npos) {
        // Find last occurrence of "functions" 
        size_t last_func_pos = content.rfind("functions");
        std::string suffix = content.substr(last_func_pos);
        
        // Check if it's an incomplete pattern at the end
        if (suffix == "functions" || suffix == "func") {
            return true;
        }
    }
    
    // 2. Incomplete function call patterns (check last occurrence in content)
    size_t func_pos = content.rfind(FUNCTIONS_PREFIX);
    if (func_pos != std::string::npos) {
        // Extract the function call part from the last occurrence
        std::string func_call_part = content.substr(func_pos);
        
        // functions. (just the prefix)
        if (func_call_part == FUNCTIONS_PREFIX) return true;
        
        // functions.name (no colon)
        size_t colon_pos = func_call_part.find(':');
        if (colon_pos == std::string::npos) return true;
        
        // functions.name: (no id)
        if (func_call_part.back() == ':') return true;
        
        // functions.name:id (no opening brace)
        size_t brace_pos = func_call_part.find('{');
        if (brace_pos == std::string::npos) return true;
        
        // Incomplete JSON detection (like original healing marker approach)
        if (brace_pos != std::string::npos) {
            std::string json_part = func_call_part.substr(brace_pos);
            if (is_incomplete_json(json_part)) return true;
        }
    }
    
    // 3. Token format partials
    if (content.find(TOOL_CALLS_SECTION_BEGIN) != std::string::npos) {
        // Check if section is incomplete
        size_t end_pos = content.find(TOOL_CALLS_SECTION_END);
        if (end_pos == std::string::npos) {
            // Section not closed, check if it has incomplete calls
            if (content.find(TOOL_CALL_BEGIN) != std::string::npos) {
                size_t call_end = content.find(TOOL_CALL_END);
                if (call_end == std::string::npos) return true; // Incomplete call
            }
            return true; // Section not closed
        }
    }
    
    // 4. Mixed format detection - look for incomplete function calls after complete ones
    size_t last_complete = 0;
    while (true) {
        size_t func_pos = content.find(FUNCTIONS_PREFIX, last_complete);
        if (func_pos == std::string::npos) break;
        
        // Check if this function call is complete
        size_t brace_pos = content.find('{', func_pos);
        if (brace_pos == std::string::npos) return true; // No opening brace
        
        // Find matching closing brace
        if (!is_json_complete_from_position(content, brace_pos)) {
            return true; // Incomplete JSON
        }
        
        // Move past this function call
        size_t closing_brace = find_matching_brace(content, brace_pos);
        if (closing_brace == std::string::npos) return true;
        last_complete = closing_brace + 1;
    }
    
    return false;
}

} // namespace kimi_k2 