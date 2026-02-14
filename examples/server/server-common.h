#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include <src/llama-impl.h>
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <cinttypes>
#include <deque>



// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "base64.hpp"


#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <set>

//// increase max payload length to allow use of larger context size
//#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
//// increase backlog size to avoid connection resets for >> 1 slots
//#define CPPHTTPLIB_LISTEN_BACKLOG 512
//// increase max URI length to handle longer prompts in query string
//#define CPPHTTPLIB_REQUEST_URI_MAX_LENGTH 32768
//// disable Nagle's algorithm
//#define CPPHTTPLIB_TCP_NODELAY true
#include <cpp-httplib/httplib.h>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::ordered_json;

#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

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

using raw_buffer = std::vector<uint8_t>;

void server_log(const char* level, const char* function, int line, const char* message, const json& extra);

template <typename T>
static T json_value(const json& body, const std::string& key, const T& default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        }
        catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const& err) {
            std::stringstream ss;
            ss << "Wrong type supplied for parameter '" << key << "'. Expected '" << json(default_value).type_name() << "', using default value: " << err.what();
            LOG_WARNING(ss.str().c_str(), body);
            return default_value;
        }
    }
    else {
        return default_value;
    }
}

// Control vector container for dynamic management
struct control_vector_container {
    std::string path;
    float scale;
    int32_t layer_start;
    int32_t layer_end;
    llama_control_vector_data data;
    bool applied;
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger& value) : value(value) {}
    server_grammar_trigger(const json& in);

    json to_json() const;
};


//
// chat template utils
//

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

bool is_base64(uint8_t c);

std::vector<uint8_t> base64_decode(const std::string& encoded_string);

//
// random string / id
//

std::string random_string();

std::string gen_chatcmplid();

std::string gen_tool_call_id();

//
// other common utils
//
float get_slot_similarity(size_t lcp, size_t prompt_length, size_t cache_length);

size_t common_part(const std::vector<llama_token>& a, const std::vector<llama_token>& b);

size_t common_part(const std::string& a, const std::string& b);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// TODO: reuse common_token_to_piece

std::string tokens_to_str(llama_context* ctx, const llama_tokens& tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context* ctx, const llama_token token);


struct common_prefix {
    size_t first = 0;
    size_t second = 0;
};

common_prefix common_prefix_add(const common_prefix& a, const common_prefix& b);

common_prefix find_common_string_prefix(const std::string& a_str, const std::string& b_str, const std::set<char>& ignore_set);

size_t find_n_tokens_from_string(const llama_context* ctx, const llama_tokens& a, const size_t max_size, size_t start,
    std::vector<size_t>& map);

std::string remove_with_set(std::string str, const std::set<char>& chars_to_remove);

common_prefix find_largest_common_number(const std::vector<size_t>& a_list, const std::vector<size_t>& b_list);

size_t find_n_tokens_from_string_with_ignore(const llama_context* ctx, const llama_tokens& a, const size_t max_size, size_t start, const std::set<char>& ignore_set,
    std::vector<size_t>& map);

common_prefix find_common_text_token_prefix(const llama_context* ctx, const llama_tokens& a, const llama_tokens& b,
    size_t start, bool exact);

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;
    float prob;

    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string& str);

    static json probs_vector_to_json(const std::vector<completion_token_output>& probs, bool post_sampling_probs);
};

using completion_token_outputs = std::deque<completion_token_output>;

// convert a vector of completion_token_output to json
json probs_vector_to_json(const llama_context* ctx, const std::vector<completion_token_output>& probs);

bool server_sent_event(httplib::DataSink& sink, const json& data);

bool server_sent_oai_resp_event(httplib::DataSink& sink, const json& data);

bool server_sent_anthropic_event(httplib::DataSink& sink, const json& data);

//
// OAI utils
//
// used by /completions endpoint
json oaicompat_chat_params_parse(const json& body);

struct oaicompat_parser_options {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs;
    common_chat_templates* tmpls;
    bool allow_image;
    bool allow_audio;
    bool enable_thinking = true;
};

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    const struct llama_model* model,
    json& body, /* openai api json semantics */
    const oaicompat_parser_options& opt,
    std::vector<raw_buffer>& out_files);

// convert OpenAI Responses API format to OpenAI Chat Completions API format
json convert_responses_to_chatcmpl(const json& body);

json anthropic_params_from_json(
    const struct llama_model* model,
    const json& body_in, /* anthropic messages api json semantics */
    const oaicompat_parser_options& opt,
    std::vector<raw_buffer>& out_files);


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json& data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json& data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json& data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string>& paths, const json& js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
std::vector<llama_token> tokenize_mixed(const llama_vocab* vocab, const json& json_prompt, bool add_special, bool parse_special);

json format_tokenizer_response(const std::vector<llama_token>& tokens);

json format_detokenized_response(const std::string& content);

json format_error_response(const std::string& message, const enum error_type type);

struct token_probabilities {
    float sampled_token_p;
    std::vector<llama_token_data> cur;
};

token_probabilities get_token_probabilities(llama_context* ctx, int idx, llama_token sampled_token_id, int n_sorted);

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks& mtmd_chunks, bool has_mtmd);

    server_tokens(const llama_tokens& tokens, bool has_mtmd);

    llama_pos pos_next() const;

    int n_tokens() const {
        return tokens.size();
    }

    // for debugging
    std::string str() const;

    const mtmd::input_chunk_ptr& find_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk* chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens& tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const std::vector<llama_token>& inp_tokens);

    // for compatibility with context shift and prompt truncation
    void resize(size_t size);

    llama_token* data();

    llama_tokens::iterator begin();

    llama_tokens::iterator end();

    llama_tokens::const_iterator cbegin();

    llama_tokens::const_iterator cend();

    llama_tokens tokens_data();

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const std::vector<llama_token>& get_text_tokens() const;

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const;

    bool empty() const;

    void clear();

    void keep_first(size_t n);

    std::string detokenize(const llama_context* ctx, bool special) const;

    std::string detokenize(const llama_context* ctx, bool special, size_t start, size_t length) const;

    size_t find_n_from_tokens(const llama_context* ctx, const server_tokens& b, bool special,
        size_t start, const size_t length);

    size_t get_common_prefix_exact(const server_tokens& b) const;

    llama_tokens get_text_tokens_exclude_think(const llama_context* ctx, const thinking_tokens& think_token) const;

    common_prefix get_common_prefix(const llama_context* ctx, const server_tokens& b, bool exact = false) const;
    // take first n tokens of tokens list a
    // find the common prefix between a and b
    common_prefix get_common_prefix_first_n(const llama_context* ctx, const server_tokens& b, size_t n, bool exact = false) const;

    // make sure all text tokens are within the vocab range
    bool validate(const struct llama_context* ctx) const;

    // encode and decode the image chunk
    int32_t process_chunk(
        llama_context* ctx,
        mtmd_context* mctx,
        size_t idx,
        llama_pos pos,
        int32_t seq_id,
        size_t& n_tokens_out) const;

    // Keep the first n_keep and remove n_discard tokens from tokens
    void discard_n_tokens(int32_t n_keep, int32_t n_discard);

    // Similarity between prompt and cached
    float get_tokens_similarity(const llama_context* ctx, const server_tokens& tokens, int n_keep = 0, int n_discard = 0) const;

    // Similarity between common part and cache
    float get_cached_tokens_similarity(const llama_context* ctx, const server_tokens& tokens, int n_keep = 0, int n_discard = 0) const;
};

// Computes FNV-1a hash of the data
std::string fnv_hash(const uint8_t* data, size_t len);

server_tokens process_mtmd_prompt(mtmd_context* mctx, std::string prompt, std::vector<raw_buffer> files);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
server_tokens tokenize_input_subprompt(const llama_vocab* vocab, mtmd_context* mctx, const json& json_prompt, bool add_special, bool parse_special);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(const llama_vocab* vocab, mtmd_context* mctx, const json& json_prompt, bool add_special, bool parse_special);

// Assuming raw_buffer has .data() and .size() members
void print_files_info(const std::vector<raw_buffer>& files);

bool prompt_cache_equal(llama_context* ctx, const server_tokens& cache_tokens,
    const server_tokens& prompt_tokens, size_t start, const common_prefix& prefix);

std::string safe_json_to_str(const json& data);
