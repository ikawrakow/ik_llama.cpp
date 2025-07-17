#pragma once

#include "json.hpp"
#include <string>

using json = nlohmann::ordered_json;

//
// Function calling parsers for multiple formats
//

// Parse XML-style function calls (format: <function_calls><invoke name="func"><parameter name="param">value</parameter></invoke></function_calls>)
static json parse_xml_function_calls(const std::string& text) {
    json tool_calls = json::array();
    
    // Look for function_calls section
    size_t section_start = text.find("<function_calls>");
    if (section_start == std::string::npos) {
        return tool_calls;
    }
    
    size_t section_end = text.find("</function_calls>", section_start);
    if (section_end == std::string::npos) {
        return tool_calls;
    }
    
    // Extract section content
    std::string section = text.substr(section_start + 16, section_end - section_start - 16);
    
    // Parse individual invoke blocks
    size_t pos = 0;
    int call_index = 0;
    while (pos < section.length()) {
        const std::string invoke_pattern = "<invoke name=\"";
        size_t invoke_start = section.find(invoke_pattern, pos);
        if (invoke_start == std::string::npos) break;
        
        size_t name_start = invoke_start + invoke_pattern.length();
        size_t name_end = section.find("\"", name_start);
        if (name_end == std::string::npos) break;
        
        std::string func_name = section.substr(name_start, name_end - name_start);
        
        size_t invoke_end = section.find("</invoke>", invoke_start);
        if (invoke_end == std::string::npos) break;
        
        // Extract parameters (skip past ">")
        size_t content_start = section.find(">", name_end) + 1;
        std::string invoke_content = section.substr(content_start, invoke_end - content_start);
        
        json arguments = json::object();
        const std::string param_pattern = "<parameter name=\"";
        const std::string param_end_pattern = "</parameter>";
        size_t param_start = invoke_content.find(param_pattern);
        while (param_start != std::string::npos) {
            size_t param_name_start = param_start + param_pattern.length();
            size_t param_name_end = invoke_content.find("\"", param_name_start);
            if (param_name_end == std::string::npos) break;
            
            std::string param_name = invoke_content.substr(param_name_start, param_name_end - param_name_start);
            
            size_t param_value_start = invoke_content.find(">", param_name_end) + 1;
            size_t param_value_end = invoke_content.find(param_end_pattern, param_value_start);
            if (param_value_end == std::string::npos) break;
            
            std::string param_value = invoke_content.substr(param_value_start, param_value_end - param_value_start);
            arguments[param_name] = param_value;
            
            param_start = invoke_content.find(param_pattern, param_value_end);
        }
        
        // Create tool call object
        json tool_call = {
            {"id", "call_" + std::to_string(call_index)},
            {"type", "function"},
            {"function", {
                {"name", func_name},
                {"arguments", arguments.dump()}
            }}
        };
        
        tool_calls.push_back(tool_call);
        
        pos = invoke_end + 9;
        call_index++;
    }
    
    return tool_calls;
}

// Parse anythingllm-style function calls (supports both JSON and XML variants)
static json parse_anythingllm_function_calls(const std::string& text) {
    json tool_calls = json::array();
    
    // Look for anythingllm function_calls section
    size_t section_start = text.find("<anythingllm:function_calls>");
    if (section_start == std::string::npos) {
        return tool_calls;
    }
    
    size_t section_end = text.find("</anythingllm:function_calls>", section_start);
    if (section_end == std::string::npos) {
        return tool_calls;
    }
    
    // Extract content between tags
    std::string content = text.substr(section_start + 28, section_end - section_start - 28);
    
    // Trim whitespace
    size_t start = content.find_first_not_of(" \t\n\r");
    size_t end = content.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        content = content.substr(start, end - start + 1);
    }
    
    // Try JSON format first (array of objects)
    if (!content.empty() && content[0] == '[') {
        try {
            json parsed = json::parse(content);
            if (parsed.is_array()) {
                int call_index = 0;
                for (const auto& call : parsed) {
                    if (call.contains("name") && (call.contains("parameters") || call.contains("arguments"))) {
                        // Handle both "parameters" and "arguments" fields
                        json args = call.contains("arguments") ? call["arguments"] : call["parameters"];
                        json tool_call = {
                            {"id", "call_" + std::to_string(call_index)},
                            {"type", "function"},
                            {"function", {
                                {"name", call["name"]},
                                {"arguments", args.dump()}
                            }}
                        };
                        tool_calls.push_back(tool_call);
                        call_index++;
                    }
                }
            }
        } catch (const std::exception& e) {
            // Continue to XML parsing if JSON fails
        }
    }
    
    // Try XML format (anythingllm:invoke structure)
    if (tool_calls.empty()) {
        size_t pos = 0;
        int call_index = 0;
        while (pos < content.length()) {
            size_t invoke_start = content.find("<anythingllm:invoke name=\"", pos);
            if (invoke_start == std::string::npos) break;
            
            size_t name_start = invoke_start + 26;
            size_t name_end = content.find("\"", name_start);
            if (name_end == std::string::npos) break;
            
            std::string func_name = content.substr(name_start, name_end - name_start);
            
            size_t invoke_end = content.find("</anythingllm:invoke>", invoke_start);
            if (invoke_end == std::string::npos) break;
            
            // Extract parameters from the invoke block
            std::string invoke_content = content.substr(name_end + 2, invoke_end - name_end - 2);
            
            json arguments = json::object();
            size_t param_start = invoke_content.find("<anythingllm:parameter_name name=\"");
            while (param_start != std::string::npos) {
                size_t param_name_start = param_start + 34;
                size_t param_name_end = invoke_content.find("\"", param_name_start);
                if (param_name_end == std::string::npos) break;
                
                std::string param_name = invoke_content.substr(param_name_start, param_name_end - param_name_start);
                
                size_t param_value_start = invoke_content.find(">", param_name_end) + 1;
                size_t param_value_end = invoke_content.find("</anythingllm:parameter_name>", param_value_start);
                if (param_value_end == std::string::npos) break;
                
                std::string param_value = invoke_content.substr(param_value_start, param_value_end - param_value_start);
                arguments[param_name] = param_value;
                
                param_start = invoke_content.find("<anythingllm:parameter_name name=\"", param_value_end);
            }
            
            // Create tool call object
            json tool_call = {
                {"id", "call_" + std::to_string(call_index)},
                {"type", "function"},
                {"function", {
                    {"name", func_name},
                    {"arguments", arguments.dump()}
                }}
            };
            
            tool_calls.push_back(tool_call);
            
            pos = invoke_end + 21;
            call_index++;
        }
    }
    
    return tool_calls;
}

// Parse token-style function calls (format: <|tool_calls_section_begin|>...<|tool_calls_section_end|>)
static json parse_token_function_calls(const std::string& text) {
    json tool_calls = json::array();
    
    // Look for tool calls section
    size_t section_start = text.find("<|tool_calls_section_begin|>");
    if (section_start == std::string::npos) {
        return tool_calls;
    }
    
    size_t section_end = text.find("<|tool_calls_section_end|>", section_start);
    if (section_end == std::string::npos) {
        return tool_calls;
    }
    
    // Extract section content
    std::string section = text.substr(section_start + 27, section_end - section_start - 27);
    
    // Parse individual tool calls
    size_t pos = 0;
    int call_index = 0;
    while (pos < section.length()) {
        size_t call_start = section.find("<|tool_call_begin|>", pos);
        if (call_start == std::string::npos) break;
        
        size_t call_end = section.find("<|tool_call_end|>", call_start);
        if (call_end == std::string::npos) break;
        
        std::string call_content = section.substr(call_start + 19, call_end - call_start - 19);
        
        // Parse tool call content
        size_t arg_start = call_content.find("<|tool_call_argument_begin|>");
        if (arg_start != std::string::npos) {
            std::string tool_id = call_content.substr(0, arg_start);
            std::string arguments = call_content.substr(arg_start + 28);
            
            // Extract function name from tool_id (format: functions.{name}:{idx})
            std::string func_name = "";
            size_t dot_pos = tool_id.find('.');
            size_t colon_pos = tool_id.find(':', dot_pos);
            if (dot_pos != std::string::npos && colon_pos != std::string::npos) {
                func_name = tool_id.substr(dot_pos + 1, colon_pos - dot_pos - 1);
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
        
        pos = call_end + 18;
        call_index++;
    }
    
    return tool_calls;
}

// Main function to parse function calls from text (supports multiple formats)
static json parse_kimi_k2_tool_calls(const std::string& text) {
    // Try anythingllm format first
    json anythingllm_result = parse_anythingllm_function_calls(text);
    if (!anythingllm_result.empty()) {
        return anythingllm_result;
    }
    
    // Try XML format
    json xml_result = parse_xml_function_calls(text);
    if (!xml_result.empty()) {
        return xml_result;
    }
    
    // Fall back to token format
    json token_result = parse_token_function_calls(text);
    return token_result;
}