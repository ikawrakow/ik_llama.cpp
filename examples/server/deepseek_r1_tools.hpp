#pragma once

#include "json.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

using json = nlohmann::ordered_json;

//
// DeepSeek R1 specific tool handling
// Based on original llama.cpp implementation
//

// Check if the model is DeepSeek R1 (based on common naming patterns)
inline bool is_deepseek_r1_model(const std::string & model_name) {
    if (model_name.empty()) {
        return false;
    }
    
    // Convert to lowercase for case-insensitive comparison
    std::string lower_model = model_name;
    std::transform(lower_model.begin(), lower_model.end(), lower_model.begin(), ::tolower);
    
    // Check for DeepSeek R1 patterns (more specific than general deepseek)
    return lower_model.find("deepseek-r1") != std::string::npos || 
           lower_model.find("deepseek_r1") != std::string::npos ||
           lower_model.find("deepseek r1") != std::string::npos ||
           (lower_model.find("deepseek") != std::string::npos && 
            (lower_model.find("-r1") != std::string::npos ||
             lower_model.find("_r1") != std::string::npos ||
             lower_model.find(" r1") != std::string::npos));
}

// Generate DeepSeek R1 tool format instructions (following original template patterns)
inline std::string deepseek_r1_tool_format_instructions() {
    return "\n\nFor function calls, use the DeepSeek R1 format:\n"
           "<｜tool▁calls▁begin｜>\n"
           "<｜tool▁call▁begin｜>\n"
           "function<｜tool▁sep｜><function_name>\n"
           "```json\n"
           "{\"arguments\": \"value\"}\n"
           "```\n"
           "<｜tool▁call▁end｜>\n"
           "<｜tool▁calls▁end｜>";
}

// Generate tools description for DeepSeek R1
inline std::string deepseek_r1_tools_description(const json & tools) {
    std::string tools_desc = "# Available Tools\n\n"
                           "You have access to the following functions. "
                           "Call them when needed to assist with the user's request.\n\n";
    
    for (const auto & tool : tools) {
        if (tool.contains("function")) {
            const auto & func = tool["function"];
            tools_desc += "**" + func["name"].get<std::string>() + "**: ";
            tools_desc += func["description"].get<std::string>() + "\n";
        }
    }
    
    return tools_desc;
}

// Inject tools into existing system message content
inline std::string deepseek_r1_inject_tools_to_system(const std::string & content, const json & tools) {
    return content + "\n\n" + deepseek_r1_tools_description(tools) + deepseek_r1_tool_format_instructions();
}

// Create a new system message with tools for DeepSeek R1
inline std::string deepseek_r1_create_system_with_tools(const json & tools) {
    std::string tools_prompt = "You are a helpful assistant with access to function calling capabilities.\n\n";
    tools_prompt += deepseek_r1_tools_description(tools);
    tools_prompt += deepseek_r1_tool_format_instructions();
    return tools_prompt;
}

// Check if tools injection is needed for DeepSeek R1
inline bool deepseek_r1_should_inject_tools(const json & tools, const std::string & model_name) {
    return !tools.empty() && tools.is_array() && is_deepseek_r1_model(model_name);
}