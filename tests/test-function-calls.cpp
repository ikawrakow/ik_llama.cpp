#include <cassert>
#include <string>
#include <iostream>

// Include the function calling parser
#include "../examples/server/function_calls.hpp"

// Test data
const std::string anythingllm_json_response = R"(I'll help you check the weather.

<anythingllm:function_calls>
[
  {
    "name": "get_weather",
    "parameters": {
      "location": "Tokyo"
    }
  }
]
</anythingllm:function_calls>

Let me get that information for you.)";

// Test for Kimi K2 format with "arguments" instead of "parameters"
const std::string kimi_k2_json_response = R"(I'll help you check the weather.

<anythingllm:function_calls>
[
  {
    "name": "get_weather",
    "arguments": {
      "location": "Tokyo"
    }
  }
]
</anythingllm:function_calls>

Let me get that information for you.)";

const std::string anythingllm_xml_response = R"(I'll help you check the weather.

<anythingllm:function_calls>
<anythingllm:invoke name="get_weather">
<anythingllm:parameter_name name="location">Tokyo</anythingllm:parameter_name>
</anythingllm:invoke>
</anythingllm:function_calls>

Let me get that information for you.)";

const std::string xml_response = R"(I'll help you check the weather.

<function_calls>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</invoke>
</function_calls>

Let me get that information for you.)";

const std::string token_response = R"(I'll help you check the weather.

<|tool_calls_section_begin|>
<|tool_call_begin|>
functions.get_weather:0<|tool_call_argument_begin|>
{"location": "Tokyo"}
<|tool_call_end|>
<|tool_calls_section_end|>

Let me get that information for you.)";

const std::string no_function_calls = R"(I can help you with that. The weather in Tokyo is usually quite pleasant this time of year.)";

// Test helper
void test_assert(bool condition, const std::string& test_name) {
    if (condition) {
        std::cout << "✅ PASS: " << test_name << std::endl;
    } else {
        std::cout << "❌ FAIL: " << test_name << std::endl;
        assert(false);
    }
}

// Test cases
void test_anythingllm_json_format() {
    json result = parse_kimi_k2_tool_calls(anythingllm_json_response);
    
    test_assert(result.is_array(), "AnythingLLM JSON: Result is array");
    test_assert(result.size() == 1, "AnythingLLM JSON: Single function call");
    
    if (result.size() > 0) {
        json tool_call = result[0];
        test_assert(tool_call.contains("id"), "AnythingLLM JSON: Has ID");
        test_assert(tool_call.contains("type"), "AnythingLLM JSON: Has type");
        test_assert(tool_call.contains("function"), "AnythingLLM JSON: Has function");
        test_assert(tool_call["type"] == "function", "AnythingLLM JSON: Correct type");
        
        json function = tool_call["function"];
        test_assert(function.contains("name"), "AnythingLLM JSON: Function has name");
        test_assert(function.contains("arguments"), "AnythingLLM JSON: Function has arguments");
        test_assert(function["name"] == "get_weather", "AnythingLLM JSON: Correct function name");
        
        // Parse arguments JSON
        std::string args_str = function["arguments"];
        json args = json::parse(args_str);
        test_assert(args["location"] == "Tokyo", "AnythingLLM JSON: Correct location argument");
    }
}

void test_anythingllm_xml_format() {
    json result = parse_kimi_k2_tool_calls(anythingllm_xml_response);
    
    test_assert(result.is_array(), "AnythingLLM XML: Result is array");
    test_assert(result.size() == 1, "AnythingLLM XML: Single function call");
    
    if (result.size() > 0) {
        json tool_call = result[0];
        test_assert(tool_call["type"] == "function", "AnythingLLM XML: Correct type");
        
        json function = tool_call["function"];
        test_assert(function["name"] == "get_weather", "AnythingLLM XML: Correct function name");
        
        // Parse arguments JSON
        std::string args_str = function["arguments"];
        json args = json::parse(args_str);
        test_assert(args["location"] == "Tokyo", "AnythingLLM XML: Correct location argument");
    }
}

void test_standard_xml_format() {
    json result = parse_kimi_k2_tool_calls(xml_response);
    
    test_assert(result.is_array(), "Standard XML: Result is array");
    test_assert(result.size() == 1, "Standard XML: Single function call");
    
    if (result.size() > 0) {
        json tool_call = result[0];
        test_assert(tool_call["type"] == "function", "Standard XML: Correct type");
        
        json function = tool_call["function"];
        test_assert(function["name"] == "get_weather", "Standard XML: Correct function name");
        
        // Parse arguments JSON
        std::string args_str = function["arguments"];
        json args = json::parse(args_str);
        test_assert(args["location"] == "Tokyo", "Standard XML: Correct location argument");
    }
}

void test_token_format() {
    json result = parse_kimi_k2_tool_calls(token_response);
    
    test_assert(result.is_array(), "Token format: Result is array");
    test_assert(result.size() == 1, "Token format: Single function call");
    
    if (result.size() > 0) {
        json tool_call = result[0];
        test_assert(tool_call["type"] == "function", "Token format: Correct type");
        
        json function = tool_call["function"];
        test_assert(function["name"] == "get_weather", "Token format: Correct function name");
        
        // Arguments should be JSON string
        std::string args_str = function["arguments"];
        json args = json::parse(args_str);
        test_assert(args["location"] == "Tokyo", "Token format: Correct location argument");
    }
}

void test_no_function_calls() {
    json result = parse_kimi_k2_tool_calls(no_function_calls);
    
    test_assert(result.is_array(), "No function calls: Result is array");
    test_assert(result.size() == 0, "No function calls: Empty array");
}

void test_multiple_function_calls() {
    std::string multiple_calls = R"(I'll help you with both tasks.

<anythingllm:function_calls>
[
  {
    "name": "get_weather",
    "parameters": {
      "location": "Tokyo"
    }
  },
  {
    "name": "calculate",
    "parameters": {
      "expression": "15 * 23"
    }
  }
]
</anythingllm:function_calls>

Here are the results.)";

    json result = parse_kimi_k2_tool_calls(multiple_calls);
    
    test_assert(result.is_array(), "Multiple calls: Result is array");
    test_assert(result.size() == 2, "Multiple calls: Two function calls");
    
    if (result.size() >= 2) {
        json first_call = result[0];
        json second_call = result[1];
        
        test_assert(first_call["function"]["name"] == "get_weather", "Multiple calls: First function name");
        test_assert(second_call["function"]["name"] == "calculate", "Multiple calls: Second function name");
    }
}

void test_malformed_input() {
    std::string malformed = R"(I'll check the weather.

<anythingllm:function_calls>
[
  {
    "name": "get_weather",
    "parameters": {
      "location": "Tokyo"
    }
  }
<!-- Missing closing tag -->

Let me help you.)";

    json result = parse_kimi_k2_tool_calls(malformed);
    
    test_assert(result.is_array(), "Malformed input: Result is array");
    test_assert(result.size() == 0, "Malformed input: Empty array for malformed input");
}

void test_kimi_k2_arguments_format() {
    json result = parse_kimi_k2_tool_calls(kimi_k2_json_response);
    
    test_assert(result.is_array(), "Kimi K2 Arguments: Result is array");
    test_assert(result.size() == 1, "Kimi K2 Arguments: Single function call");
    
    if (result.size() > 0) {
        json tool_call = result[0];
        test_assert(tool_call.contains("id"), "Kimi K2 Arguments: Has ID");
        test_assert(tool_call.contains("type"), "Kimi K2 Arguments: Has type");
        test_assert(tool_call.contains("function"), "Kimi K2 Arguments: Has function");
        test_assert(tool_call["type"] == "function", "Kimi K2 Arguments: Correct type");
        
        json function = tool_call["function"];
        test_assert(function.contains("name"), "Kimi K2 Arguments: Function has name");
        test_assert(function.contains("arguments"), "Kimi K2 Arguments: Function has arguments");
        test_assert(function["name"] == "get_weather", "Kimi K2 Arguments: Correct function name");
        
        // Parse arguments JSON
        std::string args_str = function["arguments"];
        json args = json::parse(args_str);
        test_assert(args["location"] == "Tokyo", "Kimi K2 Arguments: Correct location argument");
    }
}

int main() {
    std::cout << "🧪 Running Function Calling Parser Unit Tests" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        test_anythingllm_json_format();
        test_kimi_k2_arguments_format();
        test_anythingllm_xml_format();
        test_standard_xml_format();
        test_token_format();
        test_no_function_calls();
        test_multiple_function_calls();
        test_malformed_input();
        
        std::cout << std::endl;
        std::cout << "✅ All tests passed!" << std::endl;
        std::cout << "🚀 Function calling parser is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}