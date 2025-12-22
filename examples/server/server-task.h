#pragma once
#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};



enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

enum oaicompat_type {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
    OAICOMPAT_TYPE_EMBEDDING,
    OAICOMPAT_TYPE_ANTHROPIC,
};


struct server_task {
    int id = -1; // to be filled by server_queue
    int id_multi = -1;
    int id_target = -1;
    //int id_slot = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    server_tokens tokens;

    server_task_type type;
    json data;

    bool infill = false;
    bool embedding = false;

    server_task() = default;
    server_task(server_task_type type) : type(type) {}

};

struct result_timings {
    int32_t prompt_n = -1;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;

    int32_t predicted_n = -1;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;
    int32_t n_ctx = 0;
    int32_t n_past = 0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;

    json to_json() const;
};

struct server_task_result {
    int id = -1;
    int id_multi = -1;

    json data;

    bool stop;
    bool error;
    bool final_result = false;
    result_timings timings;
    // OAI-compat fields
    //bool                  verbose = false;
    oaicompat_type        oaicompat = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_format    oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_msg    oaicompat_msg;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs;

    int index = 0;

    std::string content;
    std::vector<llama_token> tokens;

    bool stream;
    bool include_usage;
    std::string prompt;
    //slot_params generation_params;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;

    bool post_sampling_probs = false;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    //slot_params generation_params;

    bool                  verbose = false;


    int get_index() {
        return index;
    }

    bool is_stop() {
        return true; // in stream mode, final responses are considered stop
    }

    json to_json_final();

    json to_json_partial();

    json to_json_non_oaicompat_partial();

    json to_json_non_oaicompat_final();

    json to_json_oaicompat_partial();

    json to_json_oaicompat_final();

    json to_json_oaicompat_chat_partial();

    json to_json_oaicompat_chat_final();

    json to_json_oaicompat_chat_stream();

    json to_json_anthropic_final();

    json to_json_anthropic_stream();

    json to_json_anthropic_partial();
};


struct server_prompt_checkpoint {
    llama_pos pos_min;
    llama_pos pos_max;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};


struct server_prompt {
    server_tokens tokens;
    int n_kept_prompt;
    int n_discarded_prompt;
    thinking_tokens think_tokens;

    std::vector<uint8_t> data;

    std::list<server_prompt_checkpoint> checkpoints;

    size_t size() const;

    int n_tokens() const {
        return tokens.size();
    }
};

struct server_prompt_cache {
    server_prompt_cache(llama_context* ctx, int32_t limit_size_mib, size_t limit_tokens) {
        this->ctx = ctx;
        this->limit_size = 1024ull * 1024ull * (limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;
    llama_context* ctx;
    size_t size() const;

    size_t n_tokens() const;

    server_prompt* alloc(const server_prompt& prompt, size_t state_size);

    bool load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot);

    void update();
};
