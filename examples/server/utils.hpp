#pragma once

#include "llama.h"
#include "common.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
#include "kimi_k2_tools.hpp"
#include "qwen3_tools.hpp"
#include "deepseek_r1_tools.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <random>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::ordered_json;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

enum tool_choice_type {
    TOOL_CHOICE_AUTO,
    TOOL_CHOICE_REQUIRED,
    TOOL_CHOICE_NONE,
};

extern bool server_verbose;
extern bool server_log_json;

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERB", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while (0)
#endif

#define LOG_ERROR(  MSG, ...) server_log("ERR",  __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARN", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(   MSG, ...) server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

static inline void server_log(const char * level, const char * function, int line, const char * message, const json & extra);

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            std::stringstream ss;
            ss << "Wrong type supplied for parameter '" << key << "'. Expected '" << json(default_value).type_name() << "', using default value.";
            LOG_WARNING(ss.str().c_str(), body);
            return default_value;
        }
    } else {
        return default_value;
    }
}

static inline void server_log(const char * level, const char * function, int line, const char * message, const json & extra) {
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = json{
        {"tid",       ss_tid.str()},
        {"timestamp", time(nullptr)},
    };

    if (server_log_json) {
        log.merge_patch({
            {"level",    level},
            {"function", function},
            {"line",     line},
            {"msg",      message},
        });

        if (!extra.empty()) {
            log.merge_patch(extra);
        }

        printf("%s\n", log.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    } else {
        char buf[1024];
        snprintf(buf, 1024, "%4s [%24s] %s", level, function, message);

        if (!extra.empty()) {
            log.merge_patch(extra);
        }
        std::stringstream ss;
        ss << buf << " |";
        for (const auto & el : log.items())
        {
            const std::string value = el.value().dump(-1, ' ', false, json::error_handler_t::replace);
            ss << " " << el.key() << "=" << value;
        }

        const std::string str = ss.str();
        printf("%.*s\n", (int)str.size(), str.data());
    }
    fflush(stdout);
}

//
// chat template utils
//

// Format given chat. If tmpl is empty, we take the template from model metadata
inline std::string format_chat(const struct llama_model * model, const std::string & tmpl, const std::vector<json> & messages, const json & tools = json::array(), const std::string & model_name = "") {
    std::vector<llama_chat_msg> chat;

    // Inject tools into the first system message, or create one if none exists
    bool tools_injected = false;
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & curr_msg = messages[i];

        std::string role = json_value(curr_msg, "role", std::string(""));

        std::string content;
        if (curr_msg.contains("content")) {
            if (curr_msg["content"].is_string()) {
                content = curr_msg["content"].get<std::string>();
            } else if (curr_msg["content"].is_array()) {
                for (const auto & part : curr_msg["content"]) {
                    if (part.contains("text")) {
                        content += "\n" + part["text"].get<std::string>();
                    }
                }
            } else {
                throw std::runtime_error("Invalid 'content' type (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
            }
        } else {
            throw std::runtime_error("Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
        }
        // Inject tools into the first system message, or create one if none exists
        // Only applies to Kimi-K2 models (checked by kimi_k2_should_inject_tools)
        if (kimi_k2_should_inject_tools(tools, model_name) && !tools_injected) {
            if (role == "system") {
                // Add tools to existing system message
                content = kimi_k2_inject_tools_to_system(content, tools);
                tools_injected = true;
            } else if (i == 0) {
                // Create system message with tools if no system message exists
                std::string tools_prompt = kimi_k2_create_system_with_tools(tools);
                chat.push_back({"system", tools_prompt});
                tools_injected = true;
            }
        }
        
        // Inject tools for Qwen3 models (XML Hermes format)
        if (qwen3_should_inject_tools(tools, model_name) && !tools_injected) {
            if (role == "system") {
                // Add tools to existing system message
                content = qwen3_inject_tools_to_system(content, tools);
                tools_injected = true;
            } else if (i == 0) {
                // Create system message with tools if no system message exists
                std::string tools_prompt = qwen3_create_system_with_tools(tools);
                chat.push_back({"system", tools_prompt});
                tools_injected = true;
            }
        }
        
        // Inject tools for DeepSeek R1 models
        if (deepseek_r1_should_inject_tools(tools, model_name) && !tools_injected) {
            if (role == "system") {
                // Add tools to existing system message
                content = deepseek_r1_inject_tools_to_system(content, tools);
                tools_injected = true;
            } else if (i == 0) {
                // Create system message with tools if no system message exists
                std::string tools_prompt = deepseek_r1_create_system_with_tools(tools);
                chat.push_back({"system", tools_prompt});
                tools_injected = true;
            }
        }

        chat.push_back({role, content});
    }

    auto formatted_chat = llama_chat_apply_template(model, tmpl, chat, true);
    LOG_VERBOSE("formatted_chat", {{"text", formatted_chat.c_str()}});
    return formatted_chat;
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    std::stringstream chatcmplid;
    chatcmplid << "chatcmpl-" << random_string();

    return chatcmplid.str();
}

//
// other common utils
//

static size_t common_part(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static size_t common_part(const std::string & a, const std::string & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += llama_token_to_piece(ctx, *begin);
    }

    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;

    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context * ctx, const std::vector<completion_token_output> & probs) {
    json out = json::array();

    for (const auto & prob : probs) {
        json probs_for_token = json::array();

        for (const auto & p : prob.probs) {
            const std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }

        const std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json {
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }

    return out;
}

//
// Function calling support
//
#include "function_calls.hpp"

//
// tool_choice utils
//

static tool_choice_type tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return TOOL_CHOICE_REQUIRED;
    }
    throw std::runtime_error("Invalid tool_choice: " + tool_choice);
}

//
// OAI utils
//

static json oaicompat_completion_params_parse(
    const struct llama_model * model,
    const json & body, /* openai api json semantics */
    const std::string & chat_template) {
    json llama_params;

    llama_params["__oaicompat"] = true;

    // Extract tools from the request body
    json tools = json_value(body, "tools", json::array());
    
    // Debug: Log system prompt when tools are detected
    if (!tools.empty() && server_verbose) {
        LOG_VERBOSE("Tool calls detected in request", {
            {"tool_count", tools.size()},
            {"model", json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL))}
        });
        
        // Extract and log system prompt from messages
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& msg : body["messages"]) {
                if (msg.contains("role") && msg["role"] == "system" && msg.contains("content")) {
                    std::string content_str;
                    if (msg["content"].is_string()) {
                        content_str = msg["content"];
                    } else if (msg["content"].is_array()) {
                        // Handle content blocks format
                        for (const auto& block : msg["content"]) {
                            if (block.contains("type") && block["type"] == "text" && block.contains("text")) {
                                if (!content_str.empty()) content_str += " ";
                                content_str += block["text"];
                            }
                        }
                    }
                    
                    if (!content_str.empty()) {
                        LOG_VERBOSE("System prompt with tools", {
                            {"system_prompt", content_str.substr(0, 500) + (content_str.length() > 500 ? "..." : "")}
                        });
                    }
                    break; // Only log first system message
                }
            }
        }
    }

    // Extract model name from the request body
    std::string model_name = json_value(body, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    // Apply chat template to the list of messages with tools
    llama_params["prompt"] = format_chat(model, chat_template, body.at("messages"), tools, model_name);

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            llama_params["json_schema"] = json_value(response_format, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (body.contains("logprobs")) {
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs")) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }

    // Handle tool_choice parameter
    if (body.contains("tool_choice")) {
        auto tool_choice_str = json_value(body, "tool_choice", std::string("auto"));
        auto tool_choice = tool_choice_parse_oaicompat(tool_choice_str);
        llama_params["tool_choice"] = static_cast<int>(tool_choice);
    }

    // Accept tools and tool_choice parameters for function calling support
    // Other unsupported params still rejected
    static const std::vector<std::string> unsupported_params {  };
    for (auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", "tfs_z",... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}



static json format_tokenizer_response(const std::vector<llama_token> & tokens) {
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(const std::string & content) {
    return json {
        {"content", content}
    };
}

static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}
