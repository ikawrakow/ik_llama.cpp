#pragma once

#include "json.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

using json = nlohmann::ordered_json;

//
// Kimi-K2 specific tool handling
//

// Check if the model is Kimi-K2
inline bool is_kimi_k2_model(const std::string & model_name) {
    if (model_name.empty()) {
        return false;
    }
    
    // Convert to lowercase for case-insensitive comparison
    std::string lower_model = model_name;
    std::transform(lower_model.begin(), lower_model.end(), lower_model.begin(), ::tolower);
    
    // Check if the model name contains "kimi-k2" or "kimi_k2"
    return lower_model.find("kimi-k2") != std::string::npos || 
           lower_model.find("kimi_k2") != std::string::npos;
}

// Generate Kimi-K2 tool format instructions
inline std::string kimi_k2_tool_format_instructions() {
    return "\nWhen you need to use a tool, respond with the Kimi-K2 tool call format:\n"
           "<|tool_calls_section_begin|>\n<|tool_call_begin|>\n"
           "functions.function_name:0<|tool_call_argument_begin|>\n"
           "{\"param\": \"value\"}\n"
           "<|tool_call_end|>\n<|tool_calls_section_end|>";
}

// Generate tools description for Kimi-K2
inline std::string kimi_k2_tools_description(const json & tools) {
    std::string tools_desc = "Available tools:\n";
    for (const auto & tool : tools) {
        if (tool.contains("function")) {
            const auto & func = tool["function"];
            tools_desc += "- " + func["name"].get<std::string>() + ": " + func["description"].get<std::string>() + "\n";
        }
    }
    return tools_desc;
}

// Inject tools into existing system message content
inline std::string kimi_k2_inject_tools_to_system(const std::string & content, const json & tools) {
    return content + "\n\n" + kimi_k2_tools_description(tools) + kimi_k2_tool_format_instructions();
}

// Create a new system message with tools for Kimi-K2
inline std::string kimi_k2_create_system_with_tools(const json & tools) {
    std::string tools_prompt = "You are a helpful assistant. You have access to the following tools:\n\n";
    tools_prompt += kimi_k2_tools_description(tools);
    tools_prompt += kimi_k2_tool_format_instructions();
    return tools_prompt;
}

// Check if tools injection is needed for Kimi-K2
inline bool kimi_k2_should_inject_tools(const json & tools, const std::string & model_name) {
    return !tools.empty() && tools.is_array() && is_kimi_k2_model(model_name);
} 