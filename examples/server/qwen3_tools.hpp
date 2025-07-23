#pragma once

#include "json.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

using json = nlohmann::ordered_json;

//
// Qwen3 specific tool handling (using Hermes XML format)
// Based on original llama.cpp Qwen-Qwen3-0.6B.jinja template
//

// Check if the model is Qwen3
inline bool is_qwen3_model(const std::string & model_name) {
    if (model_name.empty()) {
        return false;
    }
    
    // Convert to lowercase for case-insensitive comparison
    std::string lower_model = model_name;
    std::transform(lower_model.begin(), lower_model.end(), lower_model.begin(), ::tolower);
    
    // Check if the model name contains "qwen3" or "qwen-3"
    return lower_model.find("qwen3") != std::string::npos || 
           lower_model.find("qwen-3") != std::string::npos ||
           lower_model.find("qwen_3") != std::string::npos;
}

// Generate Qwen3 tool format instructions (XML format like Hermes)
inline std::string qwen3_tool_format_instructions() {
    return "\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
           "<tool_call>\n"
           "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
           "</tool_call>";
}

// Generate tools description for Qwen3 (XML format matching original template)
inline std::string qwen3_tools_description(const json & tools) {
    std::string tools_desc = "# Tools\n\n"
                           "You may call one or more functions to assist with the user query.\n\n"
                           "You are provided with function signatures within <tools></tools> XML tags:\n"
                           "<tools>";
    
    for (const auto & tool : tools) {
        tools_desc += "\n" + tool.dump();
    }
    
    tools_desc += "\n</tools>";
    return tools_desc;
}

// Inject tools into existing system message content
inline std::string qwen3_inject_tools_to_system(const std::string & content, const json & tools) {
    return content + "\n\n" + qwen3_tools_description(tools) + qwen3_tool_format_instructions();
}

// Create a new system message with tools for Qwen3
inline std::string qwen3_create_system_with_tools(const json & tools) {
    std::string tools_prompt = qwen3_tools_description(tools);
    tools_prompt += qwen3_tool_format_instructions();
    return tools_prompt;
}

// Check if tools injection is needed for Qwen3
inline bool qwen3_should_inject_tools(const json & tools, const std::string & model_name) {
    return !tools.empty() && tools.is_array() && is_qwen3_model(model_name);
}