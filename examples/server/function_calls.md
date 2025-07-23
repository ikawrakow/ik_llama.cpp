# Function Calling Support

This document describes the function calling format supported by the ik_llama.cpp server implementation.

## Overview

The server supports multiple native function calling formats including Kimi-K2, Qwen3 (XML), and DeepSeek R1. All function calls are automatically detected and converted to OpenAI-compatible responses.

**⚠️ Model Requirements**: Function calling support is enabled for the following model types:

- **Kimi-K2 models**: Models containing "kimi-k2" or "kimi_k2" in the model name
- **Qwen3 models**: Models containing "qwen3", "qwen-3", or "qwen_3" in the model name  
- **DeepSeek R1 models**: Models containing "deepseek-r1", "deepseek_r1", or similar patterns

Other models will not have tool injection or function call parsing enabled.

## Supported Formats

### Kimi-K2 Native Token Format

**Detection Pattern:** `<|tool_calls_section_begin|>...<|tool_calls_section_end|>`

**Structure:**
```
<|tool_calls_section_begin|>
<|tool_call_begin|>
functions.{name}:{index}<|tool_call_argument_begin|>
{JSON arguments}
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
- Native Kimi-K2 token format
- Multiple function calls supported with different indices
- Arguments are JSON objects
- Function names follow `functions.{name}:{index}` pattern

### XML-Style Format (Fallback)

**Detection Pattern:** `<tool_call>...<invoke name="...">...<parameter name="...">...</parameter>...</invoke></tool_call>`

**Structure:**
```xml
<tool_call>
<invoke name="{function_name}">
<parameter name="{param_name}">{param_value}</parameter>
<parameter name="{param_name}">{param_value}</parameter>
</invoke>
</tool_call>
```

**Example:**
```xml
<tool_call>
<invoke name="Write">
<parameter name="file_path">/path/to/file.txt</parameter>
<parameter name="content">File content here</parameter>
</invoke>
</tool_call>
```

**Notes:**
- XML-style format as fallback when model generates this format instead of token format
- Parameters are extracted as key-value pairs
- Automatically converted to JSON arguments

### DeepSeek R1 Native Format

**Detection Pattern:** `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`

**Structure:**
```
<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
function<｜tool▁sep｜>{function_name}
```json
{JSON arguments}
```
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

**Example:**
```
<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo"}
```
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>
```

**Notes:**
- Native DeepSeek R1 format ported from original llama.cpp
- Supports reasoning with `<think>...</think>` tags (automatically extracted)
- Multiple function calls supported with separate call blocks
- JSON arguments are contained within markdown code blocks

## OpenAI-Compatible Output

The native format is converted to the standard OpenAI function calling response:

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
            "id": "functions.get_weather:0",
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

### Content Filtering

When function calls are detected:
- Function call syntax is removed from content
- Tool calls are extracted into separate array
- Content is cleaned for display

### Error Handling

- Missing tokens in format returns empty array
- Malformed structure returns empty array
- Parser gracefully handles invalid JSON in arguments

## Usage with Tools Parameter

To enable function calling, include the `tools` parameter in your request:

```json
{
  "model": "kimi-k2",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Tokyo?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

## Model Compatibility

- **Kimi-K2 models**: Native support with token format
- **Qwen3 models**: Native support with XML format (Hermes-style)
- **DeepSeek R1 models**: Native support with reasoning and function call format (ported from original llama.cpp)
- **Other models**: No function calling support

## Testing

Test files are provided to verify function calling:
- `test-function-calls.cpp` - Unit tests for the native Kimi-K2 format
  - Tests native token format parsing
  - Tests multiple function calls
  - Tests error handling and malformed input

## File Structure

- `function_calls.hpp` - Parser implementation for native Kimi-K2 format
- `utils.hpp` - Integration with server (includes function_calls.hpp)
- `server.cpp` - Response formatting and content filtering