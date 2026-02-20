#include "server-task.h"
#include "server-queue.h"
#include "speculative.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <memory>



enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_PROCESSING,
};



enum slot_command {
    SLOT_COMMAND_NONE,
    SLOT_COMMAND_LOAD_PROMPT,
    SLOT_COMMAND_RELEASE,
};

struct server_slot {
    int id;
    int id_task = -1;
    int id_multi = -1;

    struct slot_params params;

    llama_batch batch_spec = {};
    llama_context * ctx_dft = nullptr;

    slot_state state = SLOT_STATE_IDLE;
    slot_command command = SLOT_COMMAND_NONE;

    llama_context* ctx = nullptr;
    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    std::unique_ptr<const server_task> task;

    // generation props
    int32_t n_ctx = 0;  // context size per slot
    int32_t n_past = 0;
    int32_t n_past_prompt = 0;
    int32_t n_decoded = 0;
    int32_t n_remaining = -1;
    int32_t n_discarded_prompt = 0;
    int32_t n_kept_prompt = 0;

    int32_t i_batch = -1;
    int32_t n_predict = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt; // can be either a string, array of strings or array of token ids

    // when a task is submitted, we first tokenize the prompt and store it here
    server_tokens prompt_tokens;
    server_tokens cache_tokens;

    std::string generated_text;

    // idx of draft tokens in the main batch
    // non-empty if we went to evaluate draft tokens
    // ref: https://github.com/ggml-org/llama.cpp/pull/17808
    std::vector<int32_t> i_batch_dft;

    std::vector<completion_token_output> generated_token_probs;
    common_chat_msg chat_msg;

    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    bool oaicompat = false;

    std::string oaicompat_model;
    std::string stopping_word;
    stop_type stop;

    // For context rewind/ token buffer
    size_t n_buffer = 0;
    int32_t rewind_count = 0;
    bool rewind_status = false;
    std::unordered_map<llama_token, float> logit_bias;
    std::vector<std::string>ban_phrases;
    completion_token_outputs token_buffer;
    float ban_phrases_bias = 0;
    int32_t banned_n = 1;

    server_prompt server_cached_prompt;

    void prompt_save(server_prompt_cache& prompt_cache) const;

    void prompt_load(server_prompt_cache& prompt_cache, const server_tokens& tokens);

    // sampling
    llama_token sampled; // in speculative mode, this is the last accepted token
    llama_tokens drafted;

    json json_schema;

    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::vector<std::string> generated_tool_call_ids;

    bool anthropic_thinking_block_started = false;
    bool anthropic_text_block_started = false;

    bool oai_resp_thinking_block_started = false;
    bool oai_resp_text_block_started = false;

    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    // multimodal
    mtmd_context* mctx = nullptr;

    // speculative decoding
    struct common_speculative * spec = nullptr;
    struct common_params_sampling sparams;
    common_sampler * ctx_sampling = nullptr;

    // speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    int32_t n_past_se = 0; // self-extend

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    void reset();

    bool has_budget(gpt_params& global_params);

    bool available() const;

    bool is_processing() const;

    void add_token_string(const completion_token_output& token);

    bool can_speculate() const;

    int get_n_draft_max() const;

    void release();

    json get_formated_timings() const;

    result_timings get_timings() const;

    const common_chat_msg& update_chat_msg(std::vector<common_chat_msg_diff>& diffs);

    size_t find_stopping_strings(const std::string& text, const size_t last_token_size, bool is_full_stop);

    void print_timings() const;

};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total = 0;
    uint64_t n_tokens_predicted_total = 0;
    uint64_t t_tokens_generation_total = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing = 0;

    uint64_t n_tokens_predicted = 0;
    uint64_t t_tokens_generation = 0;

    void init();

    void on_prompt_eval(const server_slot& slot);

    void on_prediction(const server_slot& slot);

    void reset_bucket();
};

struct server_context {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::vector<llama_lora_adapter_container> lora_adapters;
    std::vector<control_vector_container> control_vectors;

    gpt_params params_base;

    llama_batch batch;

    bool clean_kv_cache = true;
    bool add_bos_token = true;
    bool has_eos_token = false;

    // multimodal
    mtmd_context* mctx = nullptr;

    // For speculative decoding
    llama_model* model_draft = nullptr;
    llama_context* ctx_draft = nullptr;
    llama_context_params cparams_dft;

    int32_t n_ctx; // total context for all clients / slots

    // system prompt
    bool system_need_update = false;

    std::string              system_prompt;
    std::vector<llama_token> system_tokens;

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    server_queue    queue_tasks;
    server_response queue_results;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    common_chat_templates_ptr chat_templates;
    oaicompat_parser_options  oai_parser_opt;
    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;
    int32_t cache_ram_n_min = 0;
    float cache_ram_similarity = 0.5f;

    ~server_context();

    bool load_model(const gpt_params& params_);

    void init();

    std::vector<llama_token> tokenize(const json& json_prompt, bool add_special) const;

    server_slot* get_slot_by_id(int id);

    float calculate_slot_f_keep(const server_slot& slot, llama_context* ctx, const server_tokens& a, const server_tokens& b);

    std::pair<common_prefix, float> calculate_slot_similarity(const server_slot& slot, llama_context* ctx, const server_tokens& a, const server_tokens& b);

    void copy_data_to_cached_prompt(const server_tokens& tokens, server_slot& slot);

    server_slot* get_available_slot(const server_task& task);

    bool launch_slot_with_task(server_slot& slot, server_task& task);

    void kv_cache_clear();

    void system_prompt_update();

    bool system_prompt_set(const std::string& sys_prompt);

    bool process_token(completion_token_output& result, server_slot& slot);

    void populate_token_probs(const server_slot& slot, completion_token_output& result, bool post_sampling, bool special, int idx);

    json get_formated_generation(const server_slot& slot) const;

    void send_error(const server_task& task, const std::string& error, const enum error_type type = ERROR_TYPE_SERVER);

    void send_error(const server_slot& slot, const std::string& error, const enum error_type type = ERROR_TYPE_SERVER);

    void send_error(const int id_task, const int id_multi, const std::string& error, const enum error_type type = ERROR_TYPE_SERVER);

    // if multimodal is enabled, send an error and return false
    bool ensure_no_mtmd(const int id_task);

    void send_partial_response(server_slot& slot, completion_token_output tkn);

    void send_final_response(server_slot& slot);

    void send_embedding(const server_slot& slot, const llama_batch& batch);

    void request_completion(int id_task, int id_multi, json data, bool infill, bool embedding, server_tokens&& inputs);

    void request_cancel(int id_task);

    void split_multiprompt_task(int id_multi, server_task& multiprompt_task);

    void process_single_task(server_task&& task);

    void on_finish_multitask(const server_task_multi& multitask);

    void print_tokens(const server_tokens& prompt, const server_tokens& cache, size_t start1 = 0, size_t start2 = 0, size_t length = 10);

    // discard tokens in kv cache and cached tokens
    void discard_n_kv_and_cache_tokens(llama_context* ctx, server_slot& slot, int32_t n_keep, int32_t n_discard);

    // convert keep first few and discard next tokens in a to b
    void context_shift_find_n_tokens(llama_context* ctx, const server_tokens& a, const server_tokens& b, int32_t n_keep,
        int32_t n_discard, int32_t& n_kept, int32_t& n_discarded, bool exact = false);

    // handle context shift for prompt
    void context_shift_prompt(llama_context* ctx, server_slot& slot, bool exact = false);

    void update_slots();

    void release_slots();

    bool slots_idle();

    void context_shift();

    void add_sampled_tokens();

    void batch_pending_prompt(const int32_t n_ubatch, const int32_t n_batch,  int32_t & batch_type);

    void process_batch_tokens(int32_t & n_batch);

    void extend_context(const int32_t n_tokens);

    void speculative_decoding_accept();

    bool accept_special_token(const server_slot& slot, const llama_token token);

    bool has_next_token(const completion_token_output& result, server_slot& slot);

    void send_token_results(completion_token_outputs& results, server_slot& slot, int32_t n = 0);

    void buffer_and_check_string_ban(server_slot& slot, completion_token_output& result);

    json model_meta() const;

    // Re-aggregates all active vectors and updates the model state
    bool apply_control_vectors_internal();

};
