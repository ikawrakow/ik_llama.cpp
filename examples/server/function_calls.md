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

**Detection Pattern:** Multiple formats supported with automatic fallback

**⚠️ Critical Implementation Note:** DeepSeek R1 models generate different formats depending on context. The parser handles all variants automatically.

#### Format 1: Full Native Format (Primary)
**Pattern:** `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`
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

#### Format 2: Simplified Format (Fallback)
**Pattern:** `function<{function_name}>`
```
function<get_weather>
```json
{"location": "Tokyo"}
```
```

#### Format 3: Tools Array Format (New - July 2025)
**Pattern:** `function\n```json\n{"tools": [...]}`
```
function
```json
{
  "tools": [
    {
      "name": "get_weather",
      "arguments": {
        "location": "Tokyo"
      }
    },
    {
      "name": "Read",
      "arguments": {
        "file_path": "/path/to/file.java"
      }
    }
  ]
}
```
```

**Examples:**

Format 1 (Full):
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

Format 2 (Simplified):
```
function<Read>
```json
{"file_path": "/path/to/file.txt"}
```
```

Format 3 (Tools Array):
```
function
```json
{
  "tools": [
    {
      "name": "Read",
      "arguments": {
        "file_path": "/path/to/example/SystemProcessor.java"
      }
    },
    {
      "name": "Edit", 
      "arguments": {
        "file_path": "/path/to/file.java",
        "old_string": "old code",
        "new_string": "new code"
      }
    }
  ]
}
```
```

**Implementation Notes:**
- **Reasoning Support**: All formats support `<think>...</think>` reasoning tags (automatically extracted)
- **Multiple Tool Calls**: Format 1 & 2 use separate blocks, Format 3 uses array structure
- **Automatic Detection**: Parser tries formats in order: Format 1 → Format 2 → Format 3
- **Original llama.cpp Base**: Implementation follows original llama.cpp patterns exactly
- **Status**: Format 1 & 2 ✅ Working, Format 3 🔄 Partially implemented (needs debugging)

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

Comprehensive test suite for all supported formats:

### Unit Tests
- **File**: `tests/test-function-calls.cpp`
- **Coverage**: All supported model formats (Kimi-K2, Qwen3, DeepSeek R1)
- **Test Types**:
  - Native format parsing for each model type
  - Multiple function calls
  - Error handling and malformed input
  - Streaming and non-streaming responses
  - Content extraction and cleaning
  - OpenAI-compatible output generation

### DeepSeek R1 Specific Tests
- **Format 1 Tests**: Full native format with separators ✅
- **Format 2 Tests**: Simplified format without separators ✅ 
- **Format 3 Tests**: Tools array format 🔄 (TDD reproduction of server log failures)
- **Integration Tests**: Server-to-parser call chain verification
- **Regression Tests**: Ensure existing formats continue working

### Running Tests
```bash
# Build tests
cd build && make test-function-calls -j$(nproc)

# Run all function call tests
./bin/test-function-calls

# Run DeepSeek R1 specific tests
./bin/test-function-calls | grep -E "(DeepSeek|tool_calls_count)"

# Check Format 3 specific issues
./bin/test-function-calls | grep -A5 -B5 "Real failing format"
```

### Test Status
- **Kimi-K2**: ✅ All tests passing
- **Qwen3 XML**: ✅ All tests passing  
- **DeepSeek R1 Format 1 & 2**: ✅ All tests passing
- **DeepSeek R1 Format 3**: ❌ TDD tests show `tool_calls_count = 0` (needs debugging)

## File Structure

### Server Integration
- **`examples/server/server.cpp`** - Main server entry point, calls `parse_chat_message_incremental()`
- **`examples/server/function_calls.hpp`** - Server-side parser creation and integration
- **`examples/server/utils.hpp`** - Server utilities (includes function_calls.hpp)

### Core Parsing Engine  
- **`common/chat-parser.cpp`** - Main parser routing, delegates to model-specific parsers
- **`common/chat-parser.h`** - Parser interface and JSON parsing infrastructure
- **`common/chat.cpp`** - Model-specific parsing implementations:
  - `common_chat_parse_kimi_k2()` - Kimi-K2 native format
  - `common_chat_parse_qwen3()` - Qwen3 XML format  
  - `common_chat_parse_deepseek_r1()` - DeepSeek R1 multiple formats
  - `parse_deepseek_r1_tools_array()` - Format 3 tools array parser
- **`common/chat.h`** - Function declarations and model detection

### Testing
- **`tests/test-function-calls.cpp`** - Comprehensive unit tests for all formats
- **`tests/get-model.cpp`** - Test utilities for model loading

### Integration Flow
```
server.cpp:2832
  ↓ parse_chat_message_incremental(generated_text, false, modelname)
function_calls.hpp:94-95  
  ↓ common_chat_msg_parser.parse()
chat-parser.cpp:140
  ↓ model detection → specific parser
chat.cpp
  ↓ common_chat_parse_deepseek_r1() / kimi_k2() / qwen3()
  ↓ Format detection → regex matching → JSON parsing → tool_calls array
```

### Key Implementation Files
- **DeepSeek R1 Format 3**: `common/chat.cpp:266-299` (`parse_deepseek_r1_tools_array`)
- **Exception handling**: `common/chat.cpp:243-269` (Format 1 → 2 → 3 fallback chain)
- **Model detection**: `common/chat.cpp` (`is_deepseek_r1_model`, `is_qwen3_model`, etc.)
- **TDD tests**: `tests/test-function-calls.cpp:3156-3220` (Format 3 bug reproduction)