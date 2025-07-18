# Function Calling Support

This document describes the function calling formats supported by the ik_llama.cpp server implementation.

## Overview

The server supports multiple function calling formats to accommodate different model types and training approaches. All formats are automatically detected and converted to OpenAI-compatible responses.

## Supported Formats

### 1. AnythingLLM Format

**Detection Pattern:** `<anythingllm:function_calls>...</anythingllm:function_calls>`

The AnythingLLM format supports two variants:

#### Variant A: JSON Array Format
```
<anythingllm:function_calls>
[
  {
    "name": "function_name",
    "parameters": {
      "param1": "value1",
      "param2": "value2"
    }
  }
]
</anythingllm:function_calls>
```

#### Variant B: XML Structure Format
```
<anythingllm:function_calls>
<anythingllm:invoke name="function_name">
<anythingllm:parameter_name name="param1">value1</anythingllm:parameter_name>
<anythingllm:parameter_name name="param2">value2</anythingllm:parameter_name>
</anythingllm:invoke>
</anythingllm:function_calls>
```

**Example (JSON Array with "parameters"):**
```
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
```

**Example (JSON Array with "arguments" - Kimi-K2 format):**
```
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
```

**Example (XML Structure):**
```
<anythingllm:function_calls>
<anythingllm:invoke name="get_weather">
<anythingllm:parameter_name name="location">Tokyo</anythingllm:parameter_name>
</anythingllm:invoke>
</anythingllm:function_calls>
```

**Notes:**
- Parser tries JSON format first, falls back to XML structure
- Multiple function calls supported in both variants
- XML structure uses `anythingllm:invoke` and `anythingllm:parameter_name` tags
- **JSON format supports both "parameters" and "arguments" fields** for compatibility
- Kimi-K2 models typically use "arguments" instead of "parameters"

### 2. XML Function Calls Format

**Detection Pattern:** `<function_calls>...</function_calls>`

**Structure:**
```
<function_calls>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</function_calls>
```

**Example:**
```
<function_calls>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</invoke>
</function_calls>
```

**Notes:**
- XML-based structure similar to Claude format
- Multiple function calls supported with multiple `<invoke>` blocks
- Parameters are individual XML elements

### 3. Kimi-K2 Token Format

**Detection Pattern:** `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`

**Structure:**
```
<|tool_calls_section_begin|>
<|tool_call_begin|>
functions.function_name:index<|tool_call_argument_begin|>
{"param1": "value1", "param2": "value2"}
<|tool_call_end|>
<|tool_calls_section_end|>
```

**Example:**
```
<|tool_calls_section_begin|>
<|tool_call_begin|>
functions.get_weather:0<|tool_call_argument_begin|>
{"location": "Tokyo"}
<|tool_call_end|>
<|tool_calls_section_end|>
```

**Notes:**
- Uses special tokens for structure
- Function ID format: `functions.{name}:{index}`
- Arguments are JSON-encoded strings
- Multiple function calls supported with multiple `<|tool_call_begin|>` blocks

## OpenAI-Compatible Output

All formats are converted to the standard OpenAI function calling response:

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": "filtered_content_without_function_calls",
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Tokyo\"}"
            }
          }
        ]
      }
    }
  ]
}
```

## Implementation Details

### Parser Priority

The parser tries formats in this order:
1. **AnythingLLM format** (most common with current models)
2. **XML format** (fallback for Claude-style responses)
3. **Token format** (original Kimi-K2 specification)

### Content Filtering

When function calls are detected:
- The function call markup is removed from the displayed content
- `finish_reason` is set to `"tool_calls"`
- The `tool_calls` array is populated with parsed function calls

### Error Handling

- Invalid JSON in AnythingLLM format returns empty array
- Malformed XML structure returns empty array
- Missing tokens in token format returns empty array
- Parser gracefully degrades to next format on failure

## Usage with Tools Parameter

To enable function calling, include the `tools` parameter in your request:

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "required": ["location"],
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            }
          }
        }
      }
    }
  ]
}
```

## Model Compatibility

- **Kimi-K2 models**: 
  - Primarily use AnythingLLM JSON format with "arguments" field
  - Support all three formats depending on prompting
  - May fallback to XML or token formats
- **Generic models**: May use XML or AnythingLLM formats with "parameters" field
- **Fine-tuned models**: Typically use one specific format consistently

## Field Compatibility

The parser handles both parameter field names for maximum compatibility:

| Model Type | Field Name | Example |
|------------|------------|---------|
| Standard models | `"parameters"` | `{"name": "func", "parameters": {...}}` |
| Kimi-K2 models | `"arguments"` | `{"name": "func", "arguments": {...}}` |
| Both supported | Either field | Parser automatically detects and processes both |

## Testing

Test files are provided to verify function calling:
- `test_kimi_k2.py` - End-to-end API testing with Kimi-K2 format
- `test-function-calls.cpp` - Comprehensive unit tests for all parser functions
  - Tests AnythingLLM JSON format with "parameters" field
  - Tests AnythingLLM JSON format with "arguments" field (Kimi-K2)
  - Tests AnythingLLM XML format
  - Tests standard XML format
  - Tests Kimi-K2 token format
  - Tests error handling and malformed input

## File Structure

- `function_calls.hpp` - Parser implementations
- `utils.hpp` - Integration with server (includes function_calls.hpp)
- `server.cpp` - Response formatting and content filtering