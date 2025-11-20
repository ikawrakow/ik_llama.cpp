#pragma warning(disable : 4996)
#include "chat.h"
#include "utils.hpp"

#include "common.h"
#include "speculative.h"
#include "mtmd.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "grammar-parser.h"
#include "llama-vocab.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif

#include <nlohmann/json.hpp>
#include "index.html.gz.hpp"
#include "index_llamacpp.html.gz.hpp"
#include "loading.html.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <set>
#include <mutex>
#include <thread>
#include <signal.h>
#include <memory>
#include <random>
#include <algorithm>
#include <src/llama-impl.h>
#ifdef SQLITE3_MODERN_CPP_SUPPORT
#include <sqlite_modern_cpp.h>

struct DatabaseHandle {
    sqlite::database db;

    DatabaseHandle(const std::string& path) : db(path) {
        db << "CREATE TABLE IF NOT EXISTS sessions (key TEXT PRIMARY KEY, data TEXT)";
        db << "CREATE TABLE IF NOT EXISTS templates (key TEXT PRIMARY KEY, data TEXT)";
        db << "CREATE TABLE IF NOT EXISTS names (key TEXT PRIMARY KEY, data TEXT)";
    }
};
#endif

using json = nlohmann::ordered_json;

bool server_verbose = false;
bool server_log_json = true;



enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_PROCESSING,
};

enum slot_command {
    SLOT_COMMAND_NONE,
    SLOT_COMMAND_LOAD_PROMPT,
    SLOT_COMMAND_RELEASE,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
    SERVER_STATE_ERROR           // An error occurred, load_model failed
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

    json to_json() const {
        json base = {
            {"prompt_n",               prompt_n},
            {"prompt_ms",              prompt_ms},
            {"prompt_per_token_ms",    prompt_per_token_ms},
            {"prompt_per_second",      prompt_per_second},

            {"predicted_n",            predicted_n},
            {"predicted_ms",           predicted_ms},
            {"predicted_per_token_ms", predicted_per_token_ms},
            {"predicted_per_second",   predicted_per_second},

            {"n_ctx",           n_ctx},
            {"n_past",           n_past},
        };

        if (draft_n > 0) {
            base["draft_n"] = draft_n;
            base["draft_n_accepted"] = draft_n_accepted;
        }

        return base;
    }
};

struct server_task {
    int id        = -1; // to be filled by server_queue
    int id_multi  = -1;
    int id_target = -1;
    //int id_slot = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    server_tokens tokens;

    server_task_type type;
    json data;

    bool infill    = false;
    bool embedding = false;

    server_task() = default;
    server_task(server_task_type type) : type(type) {}

};

struct server_task_result {
    int id       = -1;
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

    json to_json_final() {
        switch (oaicompat) {
        case OAICOMPAT_TYPE_NONE:
            return to_json_non_oaicompat_final();
        case OAICOMPAT_TYPE_COMPLETION:
            return to_json_oaicompat_final();
        case OAICOMPAT_TYPE_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat_final();
        default:
            GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }

    json to_json_partial() {
        switch (oaicompat) {
        case OAICOMPAT_TYPE_NONE:
            return to_json_non_oaicompat_partial();
        case OAICOMPAT_TYPE_COMPLETION:
            return to_json_oaicompat_partial();
        case OAICOMPAT_TYPE_CHAT:
            return  to_json_oaicompat_chat_partial();
        default:
            GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }

    json to_json_non_oaicompat_partial() {
        // non-OAI-compat JSON
        json res = json{
            {"index",            index},
            {"content",          content},
            {"tokens",           tokens},
            {"stop",             false},
            {"id_slot",          id_multi},
            {"tokens_predicted", n_decoded},
            {"tokens_evaluated", n_prompt_tokens},
        };
        // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
        if (timings.prompt_n > 0) {
            res.push_back({ "timings", timings.to_json() });
        }
        if (!probs_output.empty()) {
            res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
        }
        return res;
    }

    json to_json_non_oaicompat_final() {
        json res = json{
            {"index",               index},
            {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
            {"tokens",              stream ? std::vector<llama_token> {} : tokens},
            {"id_slot",             id_multi},
            {"stop",                true},
            {"model",               oaicompat_model},
            {"tokens_predicted",    n_decoded},
            {"tokens_evaluated",    n_prompt_tokens},
            //{"generation_settings", default_generation_settings_for_props.to_json()},
            {"prompt",              prompt},
            {"has_new_line",        has_new_line},
            {"truncated",           truncated},
            //{"stop_type",           stop_type_to_str(STOP_TYPE_EOS)},
            {"stopping_word",       stopping_word},
            {"tokens_cached",       n_tokens_cached},
            {"timings",             timings.to_json()},
};
        if (!stream && !probs_output.empty()) {
            res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
        }
        return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
    }

    json to_json_oaicompat_partial() {
        std::time_t t = std::time(0);
        json logprobs = json(nullptr); // OAI default to null
        if (probs_output.size() > 0) {
            logprobs = json{
                {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }
        json res = json{
            {"choices",            json::array({
                json{
                    {"text",          content},
                    {"index",         index},
                    {"logprobs",      logprobs},
                    {"finish_reason", nullptr},
                }
            })},
            {"created",            t},
            {"model",              oaicompat_model},
            {"object",             "text_completion"},
            {"usage", json {
	            {"completion_tokens", n_decoded},
	            {"prompt_tokens",     n_prompt_tokens},
	            {"total_tokens",      n_decoded + n_prompt_tokens}
            }},
            {"id",                 oaicompat_cmpl_id}
        };

        // extra fields for debugging purposes
        if (verbose) {
            res["__verbose"] = to_json_non_oaicompat_partial();
        }
        if (timings.prompt_n >= 0) {
            res.push_back({ "timings", timings.to_json() });
        }

        return res;
    }

    json to_json_oaicompat_final() {
        std::time_t t = std::time(0);
        json logprobs = json(nullptr); // OAI default to null
        if (!stream && probs_output.size() > 0) {
            logprobs = json{
                {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }
        json finish_reason = "length";
        if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            finish_reason = "stop";
        }
        json res = json{
            {"choices",            json::array({
                json{
                    {"text",          stream ? "" : content}, // in stream mode, content is already in last partial chunk
                    {"index",         index},
                    {"logprobs",      logprobs},
                    {"finish_reason", finish_reason},
                }
            })},
            {"created",            t},
            {"model",              oaicompat_model},
            {"object",             "text_completion"},
            {"usage", json {
                {"completion_tokens", n_decoded},
                {"prompt_tokens",     n_prompt_tokens},
                {"total_tokens",      n_decoded + n_prompt_tokens}
            }},
            {"id", oaicompat_cmpl_id}
        };

        // extra fields for debugging purposes
        if (verbose) {
            res["__verbose"] = to_json_non_oaicompat_final();
        }
        if (timings.prompt_n >= 0) {
            res.push_back({ "timings", timings.to_json() });
        }

        return res;
    }

    json to_json_oaicompat_chat_partial() {
        bool first = n_decoded == 1;
        std::time_t t = std::time(0);
        json choices;

        std::vector<json> deltas;
        auto add_delta = [&](const json& delta) {
            deltas.push_back({
                {"choices", json::array({
                    json {
                        {"finish_reason", nullptr},
                        {"index", 0},
                        {"delta", delta},
                    },
                })},
                {"created", t},
                {"id", oaicompat_cmpl_id},
                {"model", oaicompat_model},
                {"object", "chat.completion.chunk"},
                {"usage", json {
                    {"completion_tokens", n_decoded},
                    {"prompt_tokens",     n_prompt_tokens},
                    {"total_tokens",      n_decoded + n_prompt_tokens},
                }},
                });
        };
        // We have to send an initial update to conform to openai behavior
        if (first) {
            add_delta({
                {"role", "assistant"},
                {"content", nullptr},
                });
        }

        for (const auto& diff : oaicompat_msg_diffs) {
            add_delta(common_chat_msg_diff_to_json_oaicompat<json>(diff));
        }

        if (!deltas.empty()) {
            GGML_ASSERT(deltas[deltas.size() - 1].at("choices").size() >= 1);

            if (probs_output.size() > 0) {
                deltas[deltas.size() - 1].at("choices").at(0)["logprobs"] = json{
                {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
                };
            }

            if (timings.prompt_n >= 0) {
                deltas[deltas.size() - 1].push_back({ "timings", timings.to_json() });
            }
        }

        return deltas;
    }

    json to_json_oaicompat_chat_final() {
        std::string finish_reason = "length";
        common_chat_msg msg;
        if (!oaicompat_msg.empty()) {
            msg = oaicompat_msg;
        }
        else {
            msg.role = "assistant";
            msg.content = content;
        }
        if (stop) {
            finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
        }


        json choice{
            {"finish_reason", finish_reason},
            {"index", 0},
            {"message", msg.to_json_oaicompat<json>()},
        };

        if (!stream && probs_output.size() > 0) {
            choice["logprobs"] = json{
                {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }

        std::time_t t = std::time(0);

        json res = json{
            {"choices",            json::array({choice})},
            {"created",            t},
            {"model",              oaicompat_model},
            {"object",             "chat.completion"},
            {"usage", json {
                {"completion_tokens", n_decoded},
                {"prompt_tokens",     n_prompt_tokens},
                {"total_tokens",      n_decoded + n_prompt_tokens}
            }},
            {"id", oaicompat_cmpl_id}
        };

        // extra fields for debugging purposes
        if (verbose) {
            res["__verbose"] = to_json_non_oaicompat_final();
        }
        if (timings.prompt_n >= 0) {
            res.push_back({ "timings", timings.to_json() });
        }

        return res;
    }

    json to_json_oaicompat_chat_stream() {
        std::time_t t = std::time(0);
        std::string finish_reason = "length";
        if (stop) {
        //if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
        }

        json deltas = json::array();
        for (const auto& diff : oaicompat_msg_diffs) {
            deltas.push_back({
                {"choices", json::array({
                    json {
                        {"finish_reason", nullptr},
                        {"index", 0},
                        {"delta", common_chat_msg_diff_to_json_oaicompat<json>(diff)},
                    },
                })},
                {"created", t},
                {"id", oaicompat_cmpl_id},
                {"model", oaicompat_model},
                {"object", "chat.completion.chunk"},
                });
        }

        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", finish_reason},
                    {"index", 0},
                    {"delta", json::object()},
                },
            })},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"object",             "chat.completion.chunk"},
         });
        if (include_usage) {
            // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
            // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
            deltas.push_back({
                {"choices", json::array()},
                {"created",            t},
                {"id",                 oaicompat_cmpl_id},
                {"model",              oaicompat_model},
                {"object",             "chat.completion.chunk"},
                {"usage", json {
                    {"completion_tokens", n_decoded},
                    {"prompt_tokens",     n_prompt_tokens},
                    {"total_tokens",      n_decoded + n_prompt_tokens},
                }},
                });
        }
        if (timings.prompt_n >= 0) {
            deltas.back().push_back({ "timings", timings.to_json() });
        }
        // extra fields for debugging purposes
        if (verbose && !deltas.empty()) {
            deltas.front()["__verbose"] = to_json_non_oaicompat_final();
        }

        return deltas;
    }
};

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
    case STOP_TYPE_EOS:   return "eos";
    case STOP_TYPE_WORD:  return "word";
    case STOP_TYPE_LIMIT: return "limit";
    default:              return "none";
    }
}


struct server_task_multi {
    int id = -1;

    std::set<int> subtasks_remaining;
    std::vector<server_task_result> results;
};

struct slot_params {
    bool stream       = true;
    bool include_usage = false;
    bool cache_prompt = true; // remember the prompt to avoid reprocessing all prompt

    int32_t  n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t  n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t  n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    bool timings_per_token = false;
    bool post_sampling_probs = false;
    json input_prefix;
    json input_suffix;

    // speculative decoding parameters
    struct {
        int n_max = 16;  // max drafted tokens
        int n_min = 0;  // min drafted tokens to accept
        float p_min = 0.75f; // min probability required to accept a token in the draft
    } speculative;

    // OAI-compat fields
    oaicompat_type        oaicompat = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_syntax           oaicompat_chat_syntax;

};


inline std::string get_model_name(std::string path)
{
    std::string filename = path.substr(path.find_last_of("/\\") + 1);
    return filename;
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

    std::vector<uint8_t> data;

    std::list<server_prompt_checkpoint> checkpoints;

    size_t size() const {
        size_t res = data.size();

        for (const auto& checkpoint : checkpoints) {
            res += checkpoint.size();
        }

        return res;
    }

    int n_tokens() const {
        return tokens.size();
    }
};

struct server_prompt_cache {
    server_prompt_cache(llama_context * ctx,int32_t limit_size_mib, size_t limit_tokens) {
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
    size_t size() const {
        size_t res = 0;

        for (const auto& state : states) {
            res += state.size();
        }

        return res;
    }

    size_t n_tokens() const {
        size_t res = 0;

        for (const auto& state : states) {
            res += state.n_tokens();
        }
        return res;
    }

    server_prompt* alloc(const server_prompt& prompt, size_t state_size) {
        for (auto it = states.begin(); it != states.end();) {
            auto tokens_ctx_shift = server_tokens(prompt.tokens.get_text_tokens(), false); // copy cache tokens
            tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);
            auto prefix  = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
            const size_t len = prefix.first;
            const size_t len_prompt = prefix.second;
            // first check if the current state is contained fully in the cache
            if (len_prompt == tokens_ctx_shift.size()) {
                LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
                return nullptr;
            }
            // next, remove any cached prompts that are fully contained in the current prompt
            else if(len == it->tokens.size()) {
                LLAMA_LOG_INFO(" - removing obsolete cached prompt with length %d\n", (int)len);
                it = states.erase(it);
            }
            else {
                ++it;
            }
        }

        std::vector<uint8_t> state_data;

        // check if we can allocate enough memory for the new state
        try {
            state_data.resize(state_size);
        }
        catch (const std::bad_alloc& e) {
            LLAMA_LOG_INFO("failed to allocate memory for prompt cache state: %s\n", e.what());

            limit_size = std::max<size_t>(1, 0.4 * size());

            LLAMA_LOG_INFO(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

            update();

            return nullptr;
        }

        // TODO: for some reason we can't copy server_tokens, so we have to do this workaround
        auto& cur = states.emplace_back();
        cur = {
            /*.tokens          =*/ server_tokens(prompt.tokens.get_text_tokens(), false),
            /*.n_keep          =*/ prompt.n_kept_prompt,
            /*.n_discarded_prompt     =*/ prompt.n_discarded_prompt,
            /*.data            =*/ std::move(state_data),
            /*.checkpoints     =*/ prompt.checkpoints,
        };

        return &cur;
    }

    bool load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot) {
        const auto lcp_best = prompt.tokens.get_common_prefix(ctx, tokens_new);

        float f_keep_best = float(lcp_best.second) / prompt.tokens.size();
        float sim_best = prompt.tokens.get_tokens_similarity(ctx, tokens_new, prompt.n_kept_prompt, prompt.n_discarded_prompt);
        LLAMA_LOG_INFO(" - looking for better prompt, base f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, prompt.n_kept_prompt, prompt.n_discarded_prompt);

        auto it_best = states.end();

        // find the most similar cached prompt, that would also preserve the most context
        for (auto it = states.begin(); it != states.end(); ++it) {
            const auto lcp_cur = it->tokens.get_common_prefix(ctx, tokens_new);
            const float f_keep_cur = float(lcp_cur.first) / it->tokens.size();
            const float sim_cur = it->tokens.get_tokens_similarity(ctx, tokens_new, it->n_kept_prompt, it->n_discarded_prompt);
            if (sim_best < sim_cur) {
                f_keep_best = f_keep_cur;
                sim_best = sim_cur;
                it_best = it;
            }
        }

        if (it_best != states.end()) {
            LLAMA_LOG_INFO(" - found better prompt with f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, it_best->n_kept_prompt, it_best->n_discarded_prompt);
            const size_t size = it_best->data.size();
            const size_t n = llama_state_seq_set_data(ctx, it_best->data.data(), size, id_slot);
            if (n != size) {
                LLAMA_LOG_INFO("failed to restore state with size %zu\n", size);
                return false;
            }

            it_best->data.clear();
            it_best->data.shrink_to_fit();

            prompt = std::move(*it_best);

            states.erase(it_best);
        }

        return true;
    }

    void update() {
        if (limit_size > 0) {
            // always keep at least one state, regardless of the limits
            while (states.size() > 1 && size() > limit_size) {
                if (states.empty()) {
                    break;
                }

                LLAMA_LOG_INFO(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

                states.pop_front();
            }
        }

        // average size per token
        const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

        // dynamically increase the token limit if it can fit in the memory limit
        const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size / size_per_token) : limit_tokens;

        //if (limit_tokens > 0) {
        //
        //    while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
        //        if (states.empty()) {
        //            break;
        //        }

        //        LLAMA_LOG_INFO(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
        //            limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

        //        states.pop_front();
        //    }
        //}

        LLAMA_LOG_INFO(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

        for (const auto& state : states) {
            LLAMA_LOG_INFO("   - prompt %p: %7d tokens, %7d discarded, checkpoints: %2zu, %9.3f MiB\n",
                (const void*)&state, state.n_tokens(), state.n_discarded_prompt, state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
        }
    }
};


struct server_slot {
    int id;
    int id_task = -1;
    int id_multi = -1;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;
    slot_command command = SLOT_COMMAND_NONE;

    llama_context* ctx = nullptr;
    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    std::unique_ptr<const server_task> task;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_past_prompt = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t n_discarded_prompt =  0;
    int32_t n_kept_prompt      =  0;

    int32_t i_batch     = -1;
    int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt; // can be either a string, array of strings or array of token ids

    // when a task is submitted, we first tokenize the prompt and store it here
    server_tokens prompt_tokens;
    server_tokens cache_tokens;

    std::string generated_text;

    std::vector<completion_token_output> generated_token_probs;
    common_chat_msg chat_msg;

    bool infill         = false;
    bool embedding      = false;
    bool has_next_token = true;
    bool truncated      = false;
    bool stopped_eos    = false;
    bool stopped_word   = false;
    bool stopped_limit  = false;

    bool oaicompat = false;

    std::string oaicompat_model;
    std::string stopping_word;
    stop_type stop;

    server_prompt server_cached_prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        assert(server_cached_prompt.data.size() == 0);

        const size_t cur_size = llama_state_seq_get_size(ctx, id);

        LLAMA_LOG_INFO(" - saving prompt with length %d, total state size = %.3f MiB\n",
            (int)server_cached_prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

        auto* cur = prompt_cache.alloc(server_cached_prompt, cur_size);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data(ctx, cur->data.data(), cur_size, id);
    }

    void prompt_load(server_prompt_cache& prompt_cache, const server_tokens& tokens) {
        bool res = prompt_cache.load(server_cached_prompt, tokens, ctx, id);
        if (!res) {
            LLAMA_LOG_INFO("failed to load prompt from cache\n");
        }
    }


    // sampling
    llama_token sampled;
    struct llama_sampling_params sparams;
    llama_sampling_context * ctx_sampling = nullptr;
    json json_schema;

    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::vector<std::string> generated_tool_call_ids;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    struct llama_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    llama_batch batch_spec = {};

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

    void reset() {
        n_prompt_tokens    = 0;
        generated_text     = "";
        truncated          = false;
        stopped_eos        = false;
        stopped_word       = false;
        stopped_limit      = false;
        stopping_word      = "";
        n_past             = 0;
        n_sent_text        = 0;
        n_sent_token_probs = 0;
        infill             = false;
        ga_i               = 0;
        n_past_se          = 0;
        chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

        generated_token_probs.clear();


        // Reset speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        chat_msg = {};
        json_schema = json();
        generated_tool_call_ids.clear();

        task.reset();
    }

    bool has_budget(gpt_params &global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool available() const {
        return state == SLOT_STATE_IDLE && command == SLOT_COMMAND_NONE;
    }

    bool is_processing() const {
        return (state == SLOT_STATE_IDLE && command == SLOT_COMMAND_LOAD_PROMPT) || state == SLOT_STATE_PROCESSING;
    }

    void add_token_string(const completion_token_output & token) {
        if (command == SLOT_COMMAND_RELEASE) {
            return;
        }
        generated_token_probs.push_back(token);
    }

    void release() {
        if (state == SLOT_STATE_PROCESSING) {
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
            command = SLOT_COMMAND_RELEASE;
            task.reset();
        }
    }


    json get_formated_timings() const {
        return json {
            {"prompt_n",               n_prompt_tokens_processed},
            {"prompt_ms",              t_prompt_processing},
            {"prompt_per_token_ms",    t_prompt_processing / n_prompt_tokens_processed},
            {"prompt_per_second",      1e3 / t_prompt_processing * n_prompt_tokens_processed},

            {"predicted_n",            n_decoded},
            {"predicted_ms",           t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second",   1e3 / t_token_generation * n_decoded},

            {"n_ctx",           n_ctx},
            {"n_past",           n_past},
        };
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.prompt_n = n_prompt_tokens_processed;
        timings.prompt_ms = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n = n_decoded;
        timings.predicted_ms = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

        timings.n_ctx = n_ctx;
        timings.n_past = n_past;


        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    const common_chat_msg& update_chat_msg(std::vector<common_chat_msg_diff>& diffs) {
        auto previous_msg = chat_msg;
        auto new_msg = common_chat_parse(
            generated_text,
            /* is_partial= */ stop != STOP_TYPE_EOS,
            params.oaicompat_chat_syntax);
        if (!new_msg.empty()) {
            new_msg.ensure_tool_call_ids_set(generated_tool_call_ids, gen_tool_call_id);
            chat_msg = new_msg;
            diffs = common_chat_msg_diff::compute_diffs(previous_msg, new_msg.empty() ? previous_msg : new_msg);
        }
        //LLAMA_LOG_DEBUG("Parsing chat message: %s\n", generated_text.c_str());
        //LLAMA_LOG_DEBUG("Parsing chat message: %s\n", chat_msg.reasoning_content.c_str());
        //LLAMA_LOG_DEBUG("Parsing chat message: %s\n", chat_msg.content.c_str());
        return chat_msg;
    }


    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        size_t stop_pos = std::string::npos;

        for (const std::string & word : params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stopped_word   = true;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        char buffer[512];

        double t_token = t_prompt_processing / n_prompt_tokens_processed;
        double n_tokens_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        snprintf(buffer, 512, "prompt eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)",
                t_prompt_processing, n_prompt_tokens_processed,
                t_token, n_tokens_second);

        LOG_INFO(buffer, {
            {"id_slot",                   id},
            {"id_task",                   id_task},
            {"t_prompt_processing",       t_prompt_processing},
            {"n_prompt_tokens_processed", n_prompt_tokens_processed},
            {"t_token",                   t_token},
            {"n_tokens_second",           n_tokens_second},
        });

        t_token = t_token_generation / n_decoded;
        n_tokens_second = 1e3 / t_token_generation * n_decoded;

        snprintf(buffer, 512, "generation eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)",
                t_token_generation, n_decoded,
                t_token, n_tokens_second);

        LOG_INFO(buffer, {
            {"id_slot",            id},
            {"id_task",            id_task},
            {"t_token_generation", t_token_generation},
            {"n_decoded",          n_decoded},
            {"t_token",            t_token},
            {"n_tokens_second",    n_tokens_second},
        });

        snprintf(buffer, 512, "          total time = %10.2f ms", t_prompt_processing + t_token_generation);

        LOG_INFO(buffer, {
            {"id_slot",             id},
            {"id_task",             id_task},
            {"t_prompt_processing", t_prompt_processing},
            {"t_token_generation",  t_token_generation},
            {"t_total",             t_prompt_processing + t_token_generation},
        });
    }
};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

struct server_queue {
    int id = 0;
    bool running;

    // queues
    std::vector<server_task> queue_tasks;
    std::vector<server_task> queue_tasks_deferred;

    std::vector<server_task_multi> queue_multitasks;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task       &&)> callback_new_task;
    std::function<void(server_task_multi &)> callback_finish_multitask;
    std::function<void(void)>                callback_update_slots;

    // Add a new task to the end of the queue
    int post(server_task task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (task.id == -1) {
            task.id = id++;
            LOG_VERBOSE("new task id", {{"new_id", task.id}});
        }
        queue_tasks.push_back(std::move(task));
        condition_tasks.notify_one();
        return task.id;
    }

    // Add a new task, but defer until one slot is available
    void defer(server_task && task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        queue_tasks_deferred.push_back(std::move(task));
    }

    // Get the next id for creating anew task
    int get_new_id() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        int new_id = id++;
        LOG_VERBOSE("new task id", {{"new_id", new_id}});
        return new_id;
    }

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &&)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register function to process a multitask when it is finished
    void on_finish_multitask(std::function<void(server_task_multi&)> callback) {
        callback_finish_multitask = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Call when the state of one slot is changed
    void notify_slot_changed() {
        // move deferred tasks back to main loop
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (auto & task : queue_tasks_deferred) {
            queue_tasks.push_back(std::move(task));
        }
        queue_tasks_deferred.clear();
    }

    // end the start_loop routine
    void terminate() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        running = false;
        condition_tasks.notify_all();
    }

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    void start_loop() {
        running = true;

        while (true) {
            LOG_VERBOSE("new task may arrive", {});

            while (true) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    lock.unlock();
                    break;
                }
                server_task task = std::move(queue_tasks.front());
                queue_tasks.erase(queue_tasks.begin());
                lock.unlock();
                LOG_VERBOSE("callback_new_task", {{"id_task", task.id}});
                callback_new_task(std::move(task));
            }

            LOG_VERBOSE("update_multitasks", {});

            // check if we have any finished multitasks
            auto queue_iterator = queue_multitasks.begin();
            while (queue_iterator != queue_multitasks.end()) {
                if (queue_iterator->subtasks_remaining.empty()) {
                    // all subtasks done == multitask is done
                    server_task_multi current_multitask = *queue_iterator;
                    callback_finish_multitask(current_multitask);
                    // remove this multitask
                    queue_iterator = queue_multitasks.erase(queue_iterator);
                } else {
                    ++queue_iterator;
                }
            }

            // all tasks in the current loop is processed, slots data is now ready
            LOG_VERBOSE("callback_update_slots", {});

            callback_update_slots();

            LOG_VERBOSE("wait for new task", {});
            {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    if (!running) {
                        LOG_VERBOSE("ending start_loop", {});
                        return;
                    }
                    condition_tasks.wait(lock, [&]{
                        return (!queue_tasks.empty() || !running);
                    });
                }
            }
        }
    }

    //
    // functions to manage multitasks
    //

    // add a multitask by specifying the id of all subtask (subtask is a server_task)
    void add_multitask(int id_multi, std::vector<int> & sub_ids) {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        server_task_multi multi;
        multi.id = id_multi;
        std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
        queue_multitasks.push_back(multi);
    }

    // updatethe remaining subtasks, while appending results to multitask
    void update_multitask(int id_multi, int id_sub, server_task_result & result) {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        for (auto & multitask : queue_multitasks) {
            if (multitask.id == id_multi) {
                multitask.subtasks_remaining.erase(id_sub);
                multitask.results.push_back(result);
            }
        }
    }
};

struct server_response {
    typedef std::function<void(int, int, server_task_result &)> callback_multitask_t;
    callback_multitask_t callback_update_multitask;

    // for keeping track of all tasks waiting for the result
    std::set<int> waiting_task_ids;

    // the main result queue
    std::vector<server_task_result> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task) {
        LOG_VERBOSE("waiting for task id", {{"id_task", id_task}});

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(id_task);
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        LOG_VERBOSE("remove waiting for task id", {{"id_task", id_task}});

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(id_task);
    }

    // This function blocks the thread until there is a response for this id_task
    server_task_result recv(int id_task) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&]{
                return !queue_results.empty();
            });

            for (int i = 0; i < (int) queue_results.size(); i++) {
                if (queue_results[i].id == id_task) {
                    assert(queue_results[i].id_multi == -1);
                    server_task_result res = queue_results[i];
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }

    // Register the function to update multitask
    void on_multitask_update(callback_multitask_t callback) {
        callback_update_multitask = std::move(callback);
    }

    // Send a new result to a waiting id_task
    void send(server_task_result result) {
        LOG_VERBOSE("send new result", {{"id_task", result.id}});

        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto & id_task : waiting_task_ids) {
            // LOG_TEE("waiting task id %i \n", id_task);
            // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
            if (result.id_multi == id_task) {
                LOG_VERBOSE("callback_update_multitask", {{"id_task", id_task}});
                callback_update_multitask(id_task, result.id, result);
                continue;
            }

            if (result.id == id_task) {
                LOG_VERBOSE("queue_results.push_back", {{"id_task", id_task}});
                queue_results.push_back(result);
                condition_results.notify_all();
                return;
            }
        }
    }
};

struct server_context {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::vector<llama_lora_adapter_container> lora_adapters;

    gpt_params params;

    llama_batch batch;

    bool clean_kv_cache = true;
    bool add_bos_token  = true;
    bool has_eos_token  = false;

    // multimodal
    mtmd_context * mctx = nullptr;

    // For speculative decoding
    llama_model * model_draft = nullptr;
    llama_context * ctx_draft = nullptr;
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

    ~server_context() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }

        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
        // Free multimodal
        mtmd_free(mctx);
        // Free draft model and context if they exist
        if (ctx_draft) {
            llama_free(ctx_draft);
            ctx_draft = nullptr;
        }
        if (model_draft) {
            llama_free_model(model_draft);
            model_draft = nullptr;
        }

        // Clear any sampling context
        for (server_slot & slot : slots) {
            if (slot.ctx_sampling != nullptr) {
                llama_sampling_free(slot.ctx_sampling);
            }
            if (slot.ctx_dft) {
                llama_free(slot.ctx_dft);
            }
            if (slot.spec) {
                llama_speculative_free(slot.spec);
            }
            llama_batch_free(slot.batch_spec);
        }

        llama_batch_free(batch);
    }

    bool load_model(const gpt_params & params_) {
        params = params_;

        llama_init_result llama_init = llama_init_from_gpt_params(params);

        model = llama_init.model;
        ctx = llama_init.context;
        lora_adapters = llama_init.lora_adapters;

        if (model == nullptr) {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_should_add_bos_token(model);
        has_eos_token = llama_add_eos_token(model) != 1;

        chat_templates = common_chat_templates_init(model, params.chat_template);
        try {
            common_chat_format_example(chat_templates.get(), params.use_jinja, {});
        }
        catch (const std::exception& e) {
            LOG_WARNING("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
            chat_templates = common_chat_templates_init(model, "chatml");
        }

        bool has_draft_model = !params.model_draft.empty() || !params.draft_params.empty();
        std::string & mmproj_path = params.mmproj.path;
        if (!mmproj_path.empty()) {
            mtmd_context_params mparams = mtmd_context_params_default();
            mparams.use_gpu = params.mmproj_use_gpu;
            mparams.print_timings = false;
            mparams.n_threads = params.n_threads;
            mparams.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                LOG_ERROR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            LOG_INFO("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params.ctx_shift) {
                params.ctx_shift = false;
                LOG_WARNING("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            //if (params.n_cache_reuse) {
            //    params_base.n_cache_reuse = 0;
            //    SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            //}

            if (has_draft_model) {
                LOG_ERROR("%s\n", "err: speculative decode is not supported by multimodal");
                return false;
            }
        }
        // Load draft model for speculative decoding if specified
        if (has_draft_model) {
            LLAMA_LOG_INFO("\n\n==================================loading DRAFT model==================================\n\n");

            gpt_params params_dft;
            params_dft.devices = params.devices_draft;
            params_dft.model = params.model_draft;
            params_dft.n_gpu_layers = params.n_gpu_layers_draft;
            params_dft.cache_type_k = params.cache_type_k_draft.empty() ? params.cache_type_k : params.cache_type_k_draft;
            params_dft.cache_type_v = params.cache_type_v_draft.empty() ? params.cache_type_v : params.cache_type_v_draft;
            params_dft.flash_attn = params.flash_attn;
            if (!params.draft_params.empty()) {
                auto [argc, argv] = parse_command_line("llama-server "+params.draft_params);
                if (!gpt_params_parse(argc, argv, params_dft)) {
                    gpt_params_print_usage(argc, argv, params_dft);
                    free_command_line(argc, argv);
                    return false;
                };
                free_command_line(argc, argv);
            }
            LOG_INFO("", { {"model", params_dft.model} });
            if (params_dft.n_ctx == 0) {
                params_dft.n_ctx = params.n_ctx_draft;
            }
            params_dft.n_ctx = params_dft.n_ctx == 0 ? params.n_ctx / params.n_parallel : params_dft.n_ctx;
            params_dft.n_parallel = 1;

            llama_init_result llama_init_dft = llama_init_from_gpt_params(params_dft);

            llama_model * model_dft = llama_init_dft.model;
            if (model_dft == nullptr) {
                LOG_ERROR("failed to load draft model", {{"model", params.model_draft}});
                return false;
            }

            if (!llama_speculative_are_compatible(ctx, llama_init_dft.context)) {
                LOG_INFO("the draft model is not compatible with the target model. tokens will be translated between the draft and target models.", {{}});
            }

            const int n_ctx_dft = llama_n_ctx(llama_init_dft.context);

            cparams_dft = llama_context_params_from_gpt_params(params_dft);
            cparams_dft.n_batch = n_ctx_dft;

            model_draft = llama_init_dft.model;
            ctx_draft = llama_init_dft.context;
        }
        return true;
    }


    void init() {
        const int32_t n_ctx_slot = n_ctx / params.n_parallel;

        LOG_INFO("initializing slots", {{"n_slots", params.n_parallel}});

        for (int i = 0; i < params.n_parallel; i++) {
            server_slot slot;

            slot.id = i;
            slot.ctx = ctx;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;
            slot.mctx = mctx;
            slot.cache_tokens.has_mtmd = mctx != nullptr;

            LOG_INFO("new slot", {
                {"id_slot",    slot.id},
                {"n_ctx_slot", slot.n_ctx}
            });

            const int ga_n = params.grp_attn_n;
            const int ga_w = params.grp_attn_w;

            if (ga_n != 1) {
                GGML_ASSERT(ga_n > 0                    && "ga_n must be positive");                       // NOLINT
                GGML_ASSERT(ga_w % ga_n == 0            && "ga_w must be a multiple of ga_n");             // NOLINT
                //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of ga_w");    // NOLINT
                //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * ga_n"); // NOLINT

                LOG_INFO("slot self-extend", {
                    {"id_slot", slot.id},
                    {"ga_n",    ga_n},
                    {"ga_w",    ga_w}
                });
            }

            slot.ga_i = 0;
            slot.ga_n = ga_n;
            slot.ga_w = ga_w;

            slot.sparams = params.sparams;

            // Initialize speculative decoding if a draft model is loaded
            if (ctx_draft) {
                slot.batch_spec = llama_batch_init(slot.params.speculative.n_max + 1, 0, 1);
                // slot.ctx_dft = llama_new_context_with_model(model_draft, cparams_dft); // initialized twice
                slot.ctx_dft = ctx_draft;
                if (slot.ctx_dft == nullptr) {
                    LOG_ERROR("failed to create draft context", {});
                    return;
                }

                slot.spec = llama_speculative_init(ctx, slot.ctx_dft);
                if (slot.spec == nullptr) {
                    LOG_ERROR("failed to create speculator", {});
                    return;
                }
                for (auto & pair : params.replacements_draft) {
                    llama_speculative_add_replacement_tgt_dft(slot.spec, pair.first.c_str(), pair.second.c_str());
                }

            }

            slot.reset();

            slots.push_back(std::move(slot));
        }

        default_generation_settings_for_props = get_formated_generation(slots.front());
        default_generation_settings_for_props["seed"] = -1;

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);

            // only a single seq_id per token is needed
            batch = llama_batch_init(std::max(n_batch, params.n_parallel), 0, 1);
        }

        metrics.init();

        if (params.cache_ram_mib != 0) {
            if (params.cache_ram_mib < 0) {
                LLAMA_LOG_INFO("prompt cache is enabled, size limit: %s\n", "no limit");
            }
            else {
                LLAMA_LOG_INFO("prompt cache is enabled, size limit: %d MiB\n", params.cache_ram_mib);
            }
            LLAMA_LOG_INFO("%s", "use `--cache-ram 0` to disable the prompt cache\n");
            // only apply ram size limit. No token limit for now.
            prompt_cache = std::make_unique<server_prompt_cache>(ctx,params.cache_ram_mib, 0);
        }
        else {
            LLAMA_LOG_INFO("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }

        // thinking is enabled if:
        // 1. It's not explicitly disabled (reasoning_budget == 0)
        // 2. The chat template supports it
        const bool enable_thinking = params.use_jinja && params.reasoning_budget != 0 && common_chat_templates_support_enable_thinking(chat_templates.get());
        //LLAMA_LOG_INFO("Enable thinking? %d\n", enable_thinking);

        oai_parser_opt = {
            /* use_jinja             */ params.use_jinja,
            /* prefill_assistant     */ params.prefill_assistant,
            /* reasoning_format      */ params.reasoning_format,
            /* chat_template_kwargs  */ params.default_template_kwargs,
            /* common_chat_templates */ chat_templates.get(),
            /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
            /* allow_audio           */ mctx ? mtmd_support_audio(mctx) : false,
            /* enable_thinking       */ enable_thinking,
        };
    }

    std::vector<llama_token> tokenize(const json & json_prompt, bool add_special) const {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array()) {
            bool first = true;
            for (const auto & p : json_prompt) {
                if (p.is_string()) {
                    auto s = p.template get<std::string>();

                    std::vector<llama_token> p;
                    if (first) {
                        p = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
                        first = false;
                    } else {
                        p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }

                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                } else {
                    if (first) {
                        first = false;
                    }

                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        } else {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    server_slot * get_slot_by_id(int id) {
        for (server_slot & slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;
        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            int max_lcp_len = 0;
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (!slot.available()) {
                    continue;
                }
                const auto & cache_tokens = slot.cache_tokens;
                // skip the slot if it does not contains prompt
                if (cache_tokens.empty()) {
                    continue;
                }
                // length of the Longest Common Prefix between the current slot's prompt and the input prompt
                auto lcp_len = cache_tokens.get_common_prefix(slot.ctx,task.tokens);
                // fraction of the Longest Common Prefix length with respect to the input prompt and cached prompt length
                float sim_cur = cache_tokens.get_tokens_similarity(slot.ctx, task.tokens, 0, 0);
                // handle context shift
                if (slot.ga_n == 1 && slot.n_discarded_prompt > 0 && task.tokens.size()>=slot.n_ctx) {
                    float sim_cur_ctx_shift  = cache_tokens.get_tokens_similarity(slot.ctx, task.tokens, slot.n_kept_prompt, slot.n_discarded_prompt);
                    if (sim_cur_ctx_shift > sim_cur) {
                        sim_cur = sim_cur_ctx_shift;
                    }
                }

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;
                    max_lcp_len = lcp_len.first;
                    ret = &slot;
                }
            }
            if (ret != nullptr) {
                LOG_VERBOSE("selected slot by lcp similarity", {
                    {"id_slot", ret->id},
                    {"max_lcp_len", max_lcp_len},
                    {"similarity", sim_best},
                });
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = ggml_time_us();
            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (!slot.available()) {
                    continue;
                }
                // select the current slot if the criteria match
                if (slot.t_last_used < t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                LOG_VERBOSE("selected slot by lru", {
                    {"id_slot", ret->id},
                    {"t_last", t_last},
                });
            }
        }
        if (ret) {
            const auto& tokens = ret->cache_tokens;
            float f_keep = 0.0f;
            if (!tokens.empty()) {
                if (ret->ga_n == 1 && ret->n_discarded_prompt > 0 && task.tokens.size() >= ret->n_ctx) {
                    f_keep = tokens.get_cached_tokens_similarity(ret->ctx, task.tokens, ret->params.n_keep + add_bos_token, ret->n_discarded_prompt);
                }
                else {
                    f_keep = tokens.get_cached_tokens_similarity(ret->ctx,task.tokens, 0, 0);
                }
                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < cache_ram_similarity) {
                    update_cache = true;
                }
            }
            update_cache = update_cache && prompt_cache;
            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            // don't update the cache if the slot's context is above cache_ram_n_min
            update_cache = update_cache && tokens.size() >= cache_ram_n_min;

            // TODO: mtmd does not support prompt cache
            update_cache = update_cache && (ret->mctx == nullptr);

            LLAMA_LOG_INFO("prompt cache: cache size: %d, n_keep: %d, n_discarded_prompt: %d, cache_ram_n_min: %d, f_keep: %.2f, cache_ram_similarity: %.2f\n",
                (int)tokens.size(), ret->n_kept_prompt, ret->n_discarded_prompt, cache_ram_n_min, f_keep, cache_ram_similarity);
            if (update_cache) {
                const int64_t t_start = ggml_time_us();
                LLAMA_LOG_INFO("updating prompt cache\n");
                ret->server_cached_prompt.tokens = server_tokens(tokens.get_text_tokens(), false); // copy cache tokens
                ret->server_cached_prompt.n_discarded_prompt = ret->n_discarded_prompt;
                ret->server_cached_prompt.n_kept_prompt = ret->n_kept_prompt;
                
                ret->prompt_save(*prompt_cache);
                LLAMA_LOG_INFO("prompt cache save took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
            // has prompts saved earlier to load
            if (prompt_cache && !prompt_cache->states.empty()) {
                const int64_t t_start = ggml_time_us();
                ret->server_cached_prompt.tokens = server_tokens(tokens.get_text_tokens(), false); // copy cache tokens
                ret->server_cached_prompt.n_discarded_prompt = ret->n_discarded_prompt;
                ret->server_cached_prompt.n_kept_prompt = ret->n_kept_prompt;

                ret->prompt_load(*prompt_cache, task.tokens);
                prompt_cache->update();

                ret->cache_tokens = server_tokens(ret->server_cached_prompt.tokens.get_text_tokens(), false); // recover cache tokens
                ret->n_discarded_prompt = ret->server_cached_prompt.n_discarded_prompt;
                ret->n_kept_prompt = ret->server_cached_prompt.n_kept_prompt;

                LLAMA_LOG_INFO("prompt cache load took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }
        return ret;
    }

    bool launch_slot_with_task(server_slot & slot,  server_task & task) {
        slot_params default_params;
        // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
        llama_sampling_params default_sparams = params.sparams;
        auto & data = task.data;

        if (data.count("__oaicompat") != 0) {
            slot.oaicompat = true;
            slot.oaicompat_model = json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
        } else {
            slot.oaicompat = false;
            slot.oaicompat_model = "";
        }
        slot.params.timings_per_token = json_value(data, "timings_per_token", false);
        slot.params.stream             = json_value(data, "stream",            false);
        auto stream_opt = json_value(data, "stream_options", json::object());
        slot.params.include_usage = json_value(stream_opt, "include_usage", false);
        slot.params.cache_prompt       = json_value(data, "cache_prompt",      true);
        slot.params.n_predict          = json_value(data, "n_predict",         json_value(data, "max_tokens", default_params.n_predict));
        slot.sparams.top_k             = json_value(data, "top_k",             default_sparams.top_k);
        slot.sparams.top_p             = json_value(data, "top_p",             default_sparams.top_p);
        slot.sparams.min_p             = json_value(data, "min_p",             default_sparams.min_p);
        slot.sparams.tfs_z             = json_value(data, "tfs_z",             default_sparams.tfs_z);
        slot.sparams.typical_p         = json_value(data, "typical_p",         default_sparams.typical_p);
        slot.sparams.temp              = json_value(data, "temperature",       default_sparams.temp);
        slot.sparams.dynatemp_range    = json_value(data, "dynatemp_range",    default_sparams.dynatemp_range);
        slot.sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
        slot.sparams.xtc_probability = json_value(data, "xtc_probability", default_sparams.xtc_probability);
        slot.sparams.xtc_threshold = json_value(data, "xtc_threshold", default_sparams.xtc_threshold);
        slot.sparams.top_n_sigma = json_value(data, "top_n_sigma", default_sparams.top_n_sigma);
        slot.sparams.penalty_last_n    = json_value(data, "repeat_last_n",     default_sparams.penalty_last_n);
        slot.sparams.penalty_repeat    = json_value(data, "repeat_penalty",    default_sparams.penalty_repeat);
        slot.sparams.penalty_freq      = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
        slot.sparams.penalty_present   = json_value(data, "presence_penalty",  default_sparams.penalty_present);
        slot.sparams.dry_multiplier = json_value(data, "dry_multiplier", default_sparams.dry_multiplier);
        slot.sparams.dry_base = json_value(data, "dry_base", default_sparams.dry_base);
        slot.sparams.dry_allowed_length = json_value(data, "dry_allowed_length", default_sparams.dry_allowed_length);
        slot.sparams.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", default_sparams.dry_penalty_last_n);
        slot.sparams.mirostat          = json_value(data, "mirostat",          default_sparams.mirostat);
        slot.sparams.mirostat_tau      = json_value(data, "mirostat_tau",      default_sparams.mirostat_tau);
        slot.sparams.mirostat_eta      = json_value(data, "mirostat_eta",      default_sparams.mirostat_eta);
        slot.sparams.penalize_nl       = json_value(data, "penalize_nl",       default_sparams.penalize_nl);
        slot.params.n_keep             = json_value(data, "n_keep",            slot.params.n_keep);
        slot.params.n_discard          = json_value(data, "n_discard",         default_params.n_discard);
        slot.sparams.seed              = json_value(data, "seed",              default_sparams.seed);
        slot.sparams.n_probs           = json_value(data, "n_probs",           default_sparams.n_probs);
        slot.sparams.min_keep          = json_value(data, "min_keep",          default_sparams.min_keep);

        slot.params.post_sampling_probs = json_value(data, "post_sampling_probs", default_params.post_sampling_probs);

        // speculative decoding parameters
        slot.params.speculative.n_max = json_value(data, "speculative.n_max", params.n_draft);
        slot.params.speculative.n_min = json_value(data, "speculative.n_min", params.n_draft_min);
        slot.params.speculative.p_min = json_value(data, "speculative.p_min", params.p_draft_min);

        // Clamp speculative parameters
        slot.params.speculative.n_min = std::min(slot.params.speculative.n_max, slot.params.speculative.n_min);
        slot.params.speculative.n_min = std::max(slot.params.speculative.n_min, 0);
        slot.params.speculative.n_max = std::max(slot.params.speculative.n_max, 0);

        if (slot.sparams.penalty_last_n < -1) {
            throw std::runtime_error("Error: repeat_last_n must be >= -1");
        }

        if (slot.sparams.dry_penalty_last_n < -1) {
            throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
        }

        if (slot.sparams.penalty_last_n == -1) {
            // note: should be the slot's context and not the full context, but it's ok
            slot.sparams.penalty_last_n = llama_n_ctx(ctx);
        }

        if (slot.sparams.dry_penalty_last_n == -1) {
            slot.sparams.dry_penalty_last_n = llama_n_ctx(ctx);

        }
        if (slot.sparams.dry_base < 1.0f)
        {
            slot.sparams.dry_base = default_sparams.dry_base;
        }

        // sequence breakers for DRY
        {
            // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
            // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

            if (data.contains("dry_sequence_breakers")) {
                slot.sparams.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
                if (slot.sparams.dry_sequence_breakers.empty()) {
                    send_error(task, "Error: dry_sequence_breakers must be a non-empty array of strings", ERROR_TYPE_INVALID_REQUEST);
                    return false;
                }
            }
        }

        // process "json_schema" and "grammar"
        if (data.contains("json_schema") && !data.contains("grammar")) {
            try {
                auto schema                = json_value(data, "json_schema", json::object());
                LLAMA_LOG_DEBUG("JSON schema: %s\n", schema.dump(2).c_str());
                slot.sparams.grammar       = json_schema_to_grammar(schema);
                LLAMA_LOG_DEBUG("Converted grammar: %s\n", slot.sparams.grammar.c_str());
            }
            catch (const std::exception& e) {
                throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
            }
        }
        else {
            slot.sparams.grammar       = json_value(data, "grammar",           default_sparams.grammar);
            LLAMA_LOG_DEBUG("Grammar: %s\n", slot.sparams.grammar.c_str());
            slot.sparams.grammar_lazy = json_value(data, "grammar_lazy", default_sparams.grammar_lazy);
            LLAMA_LOG_DEBUG("Grammar lazy: %s\n", slot.sparams.grammar_lazy ? "true" : "false");
        }

        if (slot.params.cache_prompt && slot.ga_n != 1) {
            LOG_WARNING("cache_prompt is not supported with group-attention", {});
            slot.params.cache_prompt = false;
        }

        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
            // Might be better to reject the request with a 400 ?
            LOG_WARNING("Max tokens to predict exceeds server configuration", {
                {"params.n_predict", slot.params.n_predict},
                {"slot.n_predict",   slot.n_predict},
            });
            slot.params.n_predict = slot.n_predict;
        }

        // infill
        slot.params.input_prefix = json_value(data, "input_prefix", default_params.input_prefix);
        slot.params.input_suffix = json_value(data, "input_suffix", default_params.input_suffix);

        // get prompt
        if (!task.infill) {
            // maybe not needed since prompt has been tokenized?
            const auto & prompt = data.find("prompt");
            if (!slot.prompt_tokens.validate(ctx)) {
                send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            if (prompt == data.end()) {
                send_error(task, "\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            if ((prompt->is_string()) ||
                (prompt->is_array() &&  prompt->size() == 1 && prompt->at(0).is_string()) ||
                (prompt->is_array() && !prompt->empty()     && prompt->at(0).is_number_integer())) {
                slot.prompt = *prompt;
            } else if (prompt->is_array() && prompt->size() == 1 && prompt->at(0).is_array()) {
                slot.prompt = prompt->at(0);
            } else {
                send_error(task, "\"prompt\" must be a string or an array of integers", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            slot.prompt_tokens = std::move(task.tokens);
        }

        // penalize user-provided tokens
        {
            slot.sparams.penalty_prompt_tokens.clear();
            slot.sparams.use_penalty_prompt_tokens = false;

            const auto & penalty_prompt = data.find("penalty_prompt");

            if (penalty_prompt != data.end()) {
                if (penalty_prompt->is_string()) {
                    const auto penalty_prompt_string = penalty_prompt->get<std::string>();
                    slot.sparams.penalty_prompt_tokens = llama_tokenize(model, penalty_prompt_string, false);

                    if (slot.params.n_predict > 0) {
                        slot.sparams.penalty_prompt_tokens.reserve(slot.sparams.penalty_prompt_tokens.size() + slot.params.n_predict);
                    }
                    slot.sparams.use_penalty_prompt_tokens = true;

                    LOG_VERBOSE("penalty_prompt_tokens", {
                        {"id_slot", slot.id},
                        {"tokens",  slot.sparams.penalty_prompt_tokens},
                    });
                }
                else if (penalty_prompt->is_array()) {
                    const auto n_tokens = penalty_prompt->size();
                    slot.sparams.penalty_prompt_tokens.reserve(n_tokens + std::max(0, slot.params.n_predict));

                    const int n_vocab = llama_n_vocab(model);
                    for (const auto & penalty_token : *penalty_prompt) {
                        if (penalty_token.is_number_integer()) {
                            const auto tok = penalty_token.get<llama_token>();
                            if (tok >= 0 && tok < n_vocab) {
                                slot.sparams.penalty_prompt_tokens.push_back(tok);
                            }
                        }
                    }
                    slot.sparams.use_penalty_prompt_tokens = true;

                    LOG_VERBOSE("penalty_prompt_tokens", {
                        {"id_slot", slot.id},
                        {"tokens",  slot.sparams.penalty_prompt_tokens},
                    });
                }
            }
        }
        {
            auto it = data.find("chat_format");
            if (it != data.end()) {
                slot.params.oaicompat_chat_syntax.format = static_cast<common_chat_format>(it->get<int>());
                LLAMA_LOG_DEBUG("Chat format: %s\n", common_chat_format_name(slot.params.oaicompat_chat_syntax.format));
            }
            else {
                slot.params.oaicompat_chat_syntax.format = default_params.oaicompat_chat_syntax.format;
            }
            common_reasoning_format reasoning_format = params.reasoning_format;
            if (data.contains("reasoning_format")) {
                reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
            }
            slot.params.oaicompat_chat_syntax.reasoning_format = reasoning_format;
            slot.params.oaicompat_chat_syntax.reasoning_in_content = slot.params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
            slot.params.oaicompat_chat_syntax.parse_tool_calls = json_value(data, "parse_tool_calls", false);

            slot.params.oaicompat_chat_syntax.thinking_forced_open = json_value(data, "thinking_forced_open", false);
        }
        {

            const auto preserved_tokens = data.find("preserved_tokens");
            if (preserved_tokens != data.end()) {
                for (const auto& t : *preserved_tokens) {
                    auto ids = llama_tokenize(model, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        LOG("Preserved token: %d\n", ids[0]);
                        slot.sparams.preserved_tokens.insert(ids[0]);
                    }
                    else {
                        // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                        LOG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                    }
                }
            }
            const auto grammar_triggers = data.find("grammar_triggers");
            if (grammar_triggers != data.end()) {
                for (const auto& t : *grammar_triggers) {
                    server_grammar_trigger ct(t);
                    if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                        const auto& word = ct.value.value;
                        auto ids = llama_tokenize(model, word, /* add_special= */ false, /* parse_special= */ true);
                        if (ids.size() == 1) {
                            auto token = ids[0];
                            if (std::find(slot.sparams.preserved_tokens.begin(), slot.sparams.preserved_tokens.end(), (llama_token)token) == slot.sparams.preserved_tokens.end()) {
                                throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                            }
                            LOG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                            common_grammar_trigger trigger;
                            trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                            trigger.value = word;
                            trigger.token = token;
                            slot.sparams.grammar_triggers.push_back(std::move(trigger));
                        }
                        else {
                            LOG("Grammar trigger word: `%s`\n", word.c_str());
                            slot.sparams.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word });
                        }
                    }
                    else {
                        //slot.sparams.grammar_triggers.push_back(ct);
                        if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                            LLAMA_LOG_DEBUG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                        }
                        else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                            LLAMA_LOG_DEBUG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                        }
                        else {
                            throw std::runtime_error("Unknown grammar trigger type");
                        }
                        slot.sparams.grammar_triggers.emplace_back(std::move(ct.value));
                    }
                }
            }

            if (slot.sparams.grammar_lazy && slot.sparams.grammar_triggers.empty()) {
                throw std::runtime_error("Error: no triggers set for lazy grammar!");
            }
        }

        {
            slot.sparams.logit_bias.clear();

            if (json_value(data, "ignore_eos", false) && has_eos_token) {
                slot.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
            }

            const auto & logit_bias = data.find("logit_bias");
            if (logit_bias != data.end() && logit_bias->is_array()) {
                const int n_vocab = llama_n_vocab(model);
                for (const auto & el : *logit_bias) {
                    // TODO: we may want to throw errors here, in case "el" is incorrect
                    if (el.is_array() && el.size() == 2) {
                        float bias;
                        if (el[1].is_number()) {
                            bias = el[1].get<float>();
                        } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                            bias = -INFINITY;
                        } else {
                            continue;
                        }

                        if (el[0].is_number_integer()) {
                            llama_token tok = el[0].get<llama_token>();
                            if (tok >= 0 && tok < n_vocab) {
                                slot.sparams.logit_bias[tok] = bias;
                            }
                        } else if (el[0].is_string()) {
                            auto toks = llama_tokenize(model, el[0].get<std::string>(), false);
                            for (auto tok : toks) {
                                slot.sparams.logit_bias[tok] = bias;
                            }
                        }
                    }
                }
            }
        }

        {
            slot.params.antiprompt.clear();

            const auto & stop = data.find("stop");
            if (stop != data.end() && stop->is_array()) {
                for (const auto & word : *stop) {
                    if (!word.empty()) {
                        slot.params.antiprompt.push_back(word);
                    }
                }
            }
        }

        {
            const auto samplers = data.find("samplers");
            if (samplers != data.end()) {
                if (samplers->is_array()) {
                    slot.sparams.samplers_sequence = llama_sampling_types_from_names(*samplers, false);
                }
                else if (samplers->is_string()) {
                    slot.sparams.samplers_sequence = llama_sampling_types_from_chars(samplers->get<std::string>());
                }
                else {
                    slot.sparams.samplers_sequence = default_sparams.samplers_sequence;
                }
            }
        }

        {
            if (slot.ctx_sampling != nullptr) {
                llama_sampling_free(slot.ctx_sampling);
            }
            slot.ctx_sampling = llama_sampling_init(llama_get_model_vocab(model),slot.sparams);
            if (slot.ctx_sampling == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }

        slot.command = SLOT_COMMAND_LOAD_PROMPT;
        // slot.prompt_tokens.clear();

        LOG_INFO("slot is processing task", {
            {"id_slot", slot.id},
            {"id_task", slot.id_task},
        });

        return true;
    }

    void kv_cache_clear() {
        LOG_VERBOSE("clearing KV cache", {});

        // clear the entire KV cache
        llama_kv_cache_clear(ctx);
        clean_kv_cache = false;
    }

    void system_prompt_update() {
        LOG_VERBOSE("system prompt update", {
            {"system_prompt", system_prompt},
        });

        kv_cache_clear();
        system_tokens.clear();

        if (!system_prompt.empty()) {
            system_tokens = ::llama_tokenize(ctx, system_prompt, true);

            const int32_t n_batch = llama_n_batch(ctx);
            const int32_t n_tokens_prompt = system_tokens.size();

            for (int32_t i = 0; i < n_tokens_prompt; i += n_batch) {
                const int32_t n_tokens = std::min(n_batch, n_tokens_prompt - i);

                llama_batch_clear(batch);

                for (int32_t j = 0; j < n_tokens; ++j) {
                    llama_batch_add(batch, system_tokens[i + j], i + j, { 0 }, false);
                }

                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERROR("llama_decode() failed", {});
                    return;
                }
            }

            // assign the system KV cache to all parallel sequences
            for (int32_t i = 1; i <= params.n_parallel; ++i) {
                llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
            }
        }

        system_need_update = false;
    }

    bool system_prompt_set(const std::string & sys_prompt) {
        system_prompt = sys_prompt;

        LOG_VERBOSE("system prompt process", {
            {"system_prompt",  system_prompt},
        });

        // release all slots
        for (server_slot & slot : slots) {
            slot.release();
        }

        system_need_update = true;
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        if (slot.ctx_sampling->params.use_penalty_prompt_tokens && result.tok != -1) {
            // we can change penalty_prompt_tokens because it is always created from scratch each request
            slot.ctx_sampling->params.penalty_prompt_tokens.push_back(result.tok);
        }

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            }
            else if (slot.has_next_token && !llama_token_is_eog(model, result.tok)) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token_string(result);
            if (slot.params.stream) {
                send_partial_response(slot, result);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params)) {
            slot.stopped_limit  = true;
            slot.has_next_token = false;

            LOG_VERBOSE("stopped by limit", {
                {"id_slot",   slot.id},
                {"id_task",   slot.id_task},
                {"n_decoded", slot.n_decoded},
                {"n_predict", slot.params.n_predict},
            });
        }

        if (llama_token_is_eog(model, result.tok)) {
            slot.stopped_eos    = true;
            slot.has_next_token = false;

            LOG_VERBOSE("eos token found", {});
        }

        auto n_ctx_train = llama_n_ctx_train(model);
        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.ga_n == 1
                    && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
            LOG_WARNING("n_predict is not set and self-context extend is disabled."
                        " Limiting generated tokens to n_ctx_train to avoid EOS-less generation infinite loop", {
                    { "id_slot",              slot.id },
                    { "params.n_predict",     slot.params.n_predict },
                    { "slot.n_prompt_tokens", slot.n_prompt_tokens },
                    { "slot.n_decoded",       slot.n_decoded },
                    { "slot.n_predict",       slot.n_predict },
                    { "n_slots",              params.n_parallel },
                    { "slot.n_ctx",           slot.n_ctx },
                    { "n_ctx",                n_ctx },
                    { "n_ctx_train",          n_ctx_train },
                    { "ga_n",                 slot.ga_n },
                });
            slot.truncated      = true;
            slot.stopped_limit  = true;
            slot.has_next_token = false; // stop prediction
        }

        LOG_VERBOSE("next token", {
            {"id_slot",        slot.id},
            {"id_task",        slot.id_task},
            {"token",          result.tok},
            {"token_text",     tokens_to_output_formatted_string(ctx, result.tok)},
            {"has_next_token", slot.has_next_token},
            {"n_remain",       slot.n_remaining},
            {"n_decoded",      slot.n_decoded},
            {"stopped_eos",    slot.stopped_eos},
            {"stopped_word",   slot.stopped_word},
            {"stopped_limit",  slot.stopped_limit},
            {"stopping_word",  slot.stopping_word},
        });

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) {
        size_t n_probs = slot.sparams.n_probs;
        size_t n_vocab = llama_n_vocab(llama_get_model(ctx));

        if (post_sampling) {
            const auto * cur_p = llama_sampling_get_candidates(slot.ctx_sampling);
            const size_t max_probs = cur_p->size;

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(max_probs);
            for (size_t i = 0; i < std::min(max_probs, n_probs); i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    llama_detokenize(ctx, {cur_p->data[i].id}, special),
                    cur_p->data[i].p
                });
            }
        } else {
            auto&&[sampled_token_p, cur] = get_token_probabilities(ctx, idx, result.tok, n_probs);

            // set probability for sampled token
            result.prob = sampled_token_p;

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
                result.probs.push_back({
                    cur[i].id,
                    llama_detokenize(ctx, {cur[i].id}, special),
                    cur[i].p
                });
            }
        }
    }

    json get_formated_generation(const server_slot & slot) const {
        const auto eos_bias   =             slot.sparams.logit_bias.find(llama_token_eos(model));
        const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() && eos_bias->second < 0.0f && std::isinf(eos_bias->second);

        std::vector<std::string> samplers_sequence;
        samplers_sequence.reserve(slot.sparams.samplers_sequence.size());
        for (const auto & sampler_type : slot.sparams.samplers_sequence) {
            samplers_sequence.emplace_back(llama_sampling_type_to_str(sampler_type));
        }

        auto grammar_triggers = json::array();
        for (const auto& trigger : slot.sparams.grammar_triggers) {
            grammar_triggers.push_back(trigger.to_json<json>());
        }

        return json {
            {"n_ctx",                     slot.n_ctx},
            {"n_predict",                 slot.n_predict},     // Server configured n_predict
            {"model",                     params.model_alias},
            {"seed",                      slot.sparams.seed},
            {"temperature",               slot.sparams.temp},
            {"dynatemp_range",            slot.sparams.dynatemp_range},
            {"dynatemp_exponent",         slot.sparams.dynatemp_exponent},
            {"top_k",                     slot.sparams.top_k},
            {"top_p",                     slot.sparams.top_p},
            {"min_p",                     slot.sparams.min_p},
            {"tfs_z",                     slot.sparams.tfs_z},
            {"typical_p",                 slot.sparams.typical_p},
            {"repeat_last_n",             slot.sparams.penalty_last_n},
            {"repeat_penalty",            slot.sparams.penalty_repeat},
            {"presence_penalty",          slot.sparams.penalty_present},
            {"frequency_penalty",         slot.sparams.penalty_freq},
            {"penalty_prompt_tokens",     slot.sparams.penalty_prompt_tokens},
            {"use_penalty_prompt_tokens", slot.sparams.use_penalty_prompt_tokens},
            {"dry_multiplier",            slot.sparams.dry_multiplier},
            {"dry_base",                  slot.sparams.dry_base},
            {"dry_allowed_length",        slot.sparams.dry_allowed_length},
            {"dry_penalty_last_n",        slot.sparams.dry_penalty_last_n},
            {"dry_sequence_breakers",     slot.sparams.dry_sequence_breakers},
            {"mirostat",                  slot.sparams.mirostat},
            {"mirostat_tau",              slot.sparams.mirostat_tau},
            {"mirostat_eta",              slot.sparams.mirostat_eta},
            {"penalize_nl",               slot.sparams.penalize_nl},
            {"stop",                      slot.params.antiprompt},
            {"max_tokens",                slot.params.n_predict}, // User configured n_predict
            {"n_keep",                    slot.params.n_keep},
            {"n_discard",                 slot.params.n_discard},
            {"ignore_eos",                ignore_eos},
            {"stream",                    slot.params.stream},
            {"logit_bias",                slot.sparams.logit_bias},
            {"n_probs",                   slot.sparams.n_probs},
            {"min_keep",                  slot.sparams.min_keep},
            {"grammar",                   slot.sparams.grammar},
            {"grammar_triggers",          grammar_triggers},
            {"preserved_tokens",          slot.sparams.preserved_tokens},
            {"chat_format",               common_chat_format_name(slot.params.oaicompat_chat_syntax.format)},
            {"reasoning_format",          common_reasoning_format_name(slot.params.oaicompat_chat_syntax.reasoning_format)},
            {"reasoning_in_content",      slot.params.oaicompat_chat_syntax.reasoning_in_content},
            {"thinking_forced_open",      slot.params.oaicompat_chat_syntax.thinking_forced_open},
            {"samplers",                  samplers_sequence}
        };
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, task.id_multi, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.id_task, slot.id_multi, error, type);
    }

    void send_error(const int id_task, const int id_multi, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        LOG_ERROR("task error", {
            {"id_multi", id_multi},
            {"id_task", id_task},
            {"error", error},
        });

        server_task_result res;
        res.id       = id_task;
        res.id_multi = id_multi;
        res.stop     = false;
        res.error    = true;
        res.data     = format_error_response(error, type);

        queue_results.send(res);
    }

    // if multimodal is enabled, send an error and return false
    bool ensure_no_mtmd(const int id_task) {
        if (mctx) {
            int id_multi = 0;
            send_error(id_task, id_multi, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, completion_token_output tkn) {
        server_task_result res;
        res.final_result = false;
        res.id       = slot.id_task;
        res.id_multi = slot.id_multi;
        res.error    = false;
        res.stop     = false;
        res.stream = slot.params.stream;
        res.content = tkn.text_to_send;
        res.post_sampling_probs = slot.params.post_sampling_probs;
        res.oaicompat = slot.params.oaicompat;
        res.oaicompat_model = slot.params.oaicompat_model;
        res.oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;
        res.n_decoded = slot.n_decoded;
        res.n_prompt_tokens = slot.n_prompt_tokens;
        res.data     = json {
            {"content",    tkn.text_to_send},
            {"stop",       false},
            {"id_slot",    slot.id},
            {"multimodal", false}
        };
        slot.update_chat_msg(res.oaicompat_msg_diffs);

        // populate res.probs_output
        if (slot.sparams.n_probs > 0) {
            res.probs_output = {tkn}; // copy the token probs
            res.data["completion_probabilities"] = probs_vector_to_json(ctx, res.probs_output);
        }

        if (slot.oaicompat) {
            res.data["oaicompat_token_ctr"] = slot.n_decoded;
            res.data["model"] = slot.oaicompat_model;
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.params.timings_per_token) {
            res.timings = slot.get_timings();
        }
        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot& slot) {
        server_task_result res;
        res.final_result = true;
        res.id       = slot.id_task;
        res.id_multi = slot.id_multi;
        res.error    = false;
        res.stop = true; // to do: set value
        res.stream = slot.params.stream;
        res.include_usage = slot.params.include_usage;
        res.content = slot.generated_text;
        res.timings = slot.get_timings();
        res.post_sampling_probs = slot.params.post_sampling_probs;
        res.oaicompat = slot.params.oaicompat;
        res.oaicompat_model = slot.params.oaicompat_model;
        res.oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;
        res.oaicompat_msg = slot.update_chat_msg(res.oaicompat_msg_diffs);
        res.n_decoded = slot.n_decoded;
        res.n_prompt_tokens = slot.n_prompt_tokens;
        res.oaicompat_model = slot.oaicompat_model;
        res.data     = json {
            {"content",             !slot.params.stream ? slot.generated_text : ""},
            {"generated_text",      slot.generated_text},  // Always include full text for finish_reason logic
            {"id_slot",             slot.id},
            {"stop",                true},
            {"model",               params.model_alias},
            {"tokens_predicted",    slot.n_decoded},
            {"tokens_evaluated",    slot.n_prompt_tokens},
            {"generation_settings", get_formated_generation(slot)},
            {"prompt",              slot.prompt},
            {"truncated",           slot.truncated},
            {"stopped_eos",         slot.stopped_eos},
            {"stopped_word",        slot.stopped_word},
            {"stopped_limit",       slot.stopped_limit},
            {"stopping_word",       slot.stopping_word},
            {"tokens_cached",       slot.n_past},
            {"timings",             slot.get_formated_timings()},
            //{"oaicompat_chat_format",  slot.params.oaicompat_chat_format},
        };

        // populate res.probs_output
        if (slot.sparams.n_probs > 0) {
            res.probs_output = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(),
                    slot.generated_token_probs.end());
            res.data["completion_probabilities"] = probs_vector_to_json(ctx, res.probs_output);
        }

        if (slot.oaicompat) {
            res.data["oaicompat_token_ctr"] = slot.n_decoded;
            res.data["model"] = slot.oaicompat_model;
        }

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        server_task_result res;
        res.id       = slot.id_task;
        res.id_multi = slot.id_multi;
        res.error    = false;
        res.stop     = true;

        const int n_embd = llama_n_embd(model);

        std::vector<float> embd_res(n_embd, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                LOG_ERROR("failed to get embeddings", {
                    {"token",  batch.token [i]},
                        {"seq_id", batch.seq_id[i][0]}
                });

                res.data = json {
                    {"embedding", std::vector<float>(n_embd, 0.0f)},
                    {"tokens_evaluated", slot.n_prompt_tokens},
                };

                continue;
            }

            llama_embd_normalize(embd, embd_res.data(), n_embd);

            res.data = json {
                {"embedding", embd_res},
                {"tokens_evaluated", slot.n_prompt_tokens},
            };
        }

        queue_results.send(res);
    }

    void request_completion(int id_task, int id_multi, json data, bool infill, bool embedding, server_tokens && inputs) {
        server_task task;
        task.id        = id_task;
        task.id_multi  = id_multi;
        task.id_target = 0;
        task.data      = std::move(data);
        task.infill    = infill;
        task.embedding = embedding;
        task.type      = SERVER_TASK_TYPE_COMPLETION;
        task.tokens    = std::move(inputs);
        // when a completion task's prompt array is not a singleton, we split it into multiple requests
        // otherwise, it's a single-prompt task, we actually queue it
        // if there's numbers in the prompt array it will be treated as an array of tokens
        if (task.data.count("prompt") != 0 && task.data.at("prompt").size() > 1) {
            bool numbers = false;
            for (const auto & e : task.data.at("prompt")) {
                if (e.is_number()) {
                    numbers = true;
                    break;
                }
            }

            // NOTE: split_multiprompt_task() does not handle a mix of strings and numbers,
            // it will completely stall the server. I don't know where the bug for this is.
            //
            // if there are numbers, it needs to be treated like a single prompt,
            // queue_tasks handles a mix of strings and numbers just fine.
            if (numbers) {
                queue_tasks.post(std::move(task));
            } else {
                split_multiprompt_task(id_task, task);
            }
        } else {
            queue_tasks.post(std::move(task));
        }
    }

    void request_cancel(int id_task) {
        server_task task;
        task.type      = SERVER_TASK_TYPE_CANCEL;
        task.id_target = id_task;

        queue_tasks.post(std::move(task));
    }

    void split_multiprompt_task(int id_multi, server_task & multiprompt_task) {
        const int prompt_count = multiprompt_task.data.at("prompt").size();
        if (prompt_count <= 1) {
            send_error(multiprompt_task, "error while handling multiple prompts");
            return;
        }

        // generate all the ID for subtask
        std::vector<int> subtask_ids(prompt_count);
        for (int i = 0; i < prompt_count; i++) {
            subtask_ids[i] = queue_tasks.get_new_id();
        }

        // queue up the multitask so we can track its subtask progression
        queue_tasks.add_multitask(id_multi, subtask_ids);

        // add subtasks
        for (int i = 0; i < prompt_count; i++) {
            json subtask_data = multiprompt_task.data;
            subtask_data["prompt"] = subtask_data.at("prompt")[i];

            // subtasks inherit everything else (infill mode, embedding mode, etc.)
            request_completion(subtask_ids[i], id_multi, subtask_data, multiprompt_task.infill, multiprompt_task.embedding,
                std::move(multiprompt_task.tokens));
        }
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
                {
                    const int id_slot = json_value(task.data, "id_slot", -1);

                    server_slot * slot;

                    if (id_slot != -1) {
                        slot = get_slot_by_id(id_slot);
                    } else {
                        slot = get_available_slot(task);
                    }

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        LOG_VERBOSE("no slot is available", {{"id_task", task.id}});
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.data.contains("system_prompt")) {
                        std::string sys_prompt = json_value(task.data, "system_prompt", std::string());
                        system_prompt_set(sys_prompt);

                        for (server_slot & slot : slots) {
                            slot.n_past    = 0;
                            slot.n_past_se = 0;
                        }
                    }

                    slot->reset();

                    slot->id_task   = task.id;
                    slot->id_multi  = task.id_multi;
                    slot->infill    = task.infill;
                    slot->embedding = task.embedding;

                    if (!launch_slot_with_task(*slot, task)) {
                        LOG_ERROR("error while launching slot", task.data);
                        break;
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.id_task == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = get_formated_generation(slot);
                        slot_data["id"]         = slot.id;
                        slot_data["id_task"]    = slot.id_task;
                        slot_data["state"]      = slot.state;
                        slot_data["prompt"]     = slot.prompt;
                        slot_data["next_token"] = {
                            {"has_next_token", slot.has_next_token},
                            {"n_remain",       slot.n_remaining},
                            {"n_decoded",      slot.n_decoded},
                            {"stopped_eos",    slot.stopped_eos},
                            {"stopped_word",   slot.stopped_word},
                            {"stopped_limit",  slot.stopped_limit},
                            {"stopping_word",  slot.stopping_word},
                        };

                        if (slot_data["state"] == SLOT_STATE_IDLE) {
                            n_idle_slots++;
                        } else {
                            n_processing_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    LOG_INFO("slot data", {
                        {"id_task",            task.id},
                        {"n_idle_slots",       n_idle_slots},
                        {"n_processing_slots", n_processing_slots}
                    });

                    LOG_VERBOSE("slot data", {
                        {"id_task",            task.id},
                        {"n_idle_slots",       n_idle_slots},
                        {"n_processing_slots", n_processing_slots},
                        {"slots",              slots_data}
                    });

                    server_task_result res;
                    res.id       = task.id;
                    res.id_multi = task.id_multi;
                    res.stop     = true;
                    res.error    = false;
                    res.data     = {
                        { "idle",                            n_idle_slots       },
                        { "processing",                      n_processing_slots },
                        { "deferred",                        queue_tasks.queue_tasks_deferred.size() },
                        { "t_start",                         metrics.t_start},

                        { "n_prompt_tokens_processed_total", metrics.n_prompt_tokens_processed_total},
                        { "t_tokens_generation_total",       metrics.t_tokens_generation_total},
                        { "n_tokens_predicted_total",        metrics.n_tokens_predicted_total},
                        { "t_prompt_processing_total",       metrics.t_prompt_processing_total},

                        { "n_prompt_tokens_processed",       metrics.n_prompt_tokens_processed},
                        { "t_prompt_processing",             metrics.t_prompt_processing},
                        { "n_tokens_predicted",              metrics.n_tokens_predicted},
                        { "t_tokens_generation",             metrics.t_tokens_generation},

                        { "kv_cache_tokens_count",           llama_get_kv_cache_token_count(ctx)},
                        { "kv_cache_used_cells",             llama_get_kv_cache_used_cells(ctx)},

                        { "slots",                           slots_data },
                    };

                    if (json_value(task.data, "reset_bucket", false)) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(res);
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!ensure_no_mtmd(task.id)) {
                        break;
                    }
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->cache_tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.data.at("filename");
                    std::string filepath = task.data.at("filepath");

                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json {
                        { "id_slot",   id_slot },
                        { "filename",  filename },
                        { "n_saved",   token_count }, // tokens saved
                        { "n_written", nwrite },      // bytes written
                        { "timings", {
                            { "save_ms", t_save_ms }
                        } }
                    };
                    queue_results.send(result);
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!ensure_no_mtmd(task.id)) break;
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.data.at("filename");
                    std::string filepath = task.data.at("filepath");

                    slot->cache_tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->cache_tokens.resize(0);
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    slot->cache_tokens.resize(token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json {
                        { "id_slot",    id_slot },
                        { "filename",   filename },
                        { "n_restored", token_count }, // tokens restored
                        { "n_read",     nread },       // bytes read
                        { "timings", {
                            { "restore_ms", t_restore_ms }
                        } }
                    };
                    queue_results.send(result);
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!ensure_no_mtmd(task.id)) break;
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->cache_tokens.size();
                    llama_kv_cache_seq_rm(ctx, slot->id + 1, -1, -1);
                    slot->cache_tokens.clear();

                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json {
                        { "id_slot",  id_slot },
                        { "n_erased", n_erased }
                    };
                    queue_results.send(result);
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    llama_lora_adapters_apply(ctx, lora_adapters);
                    server_task_result result;
                    result.id = task.id;
                    result.stop = true;
                    result.error = false;
                    result.data = json{{ "success", true }};
                    queue_results.send(result);
                } break;
        }
    }

    void on_finish_multitask(const server_task_multi & multitask) {
        // all subtasks done == multitask is done
        server_task_result result;
        result.id    = multitask.id;
        result.stop  = true;
        result.error = false;

        // collect json results into one json result
        std::vector<json> result_jsons;
        for (const auto & subres : multitask.results) {
            result_jsons.push_back(subres.data);
            result.error = result.error && subres.error;
        }
        result.data = json {
            { "results", result_jsons }
        };

        queue_results.send(result);
    }

    void print_tokens(const server_tokens & prompt, const server_tokens& cache, size_t start1 = 0, size_t start2=0 , size_t length = 10) {
        if (cache.size() > start2) {
            LLAMA_LOG_INFO("cache : %s\n", cache.detokenize(ctx, true, start2, length).c_str());
        }
        if (prompt.size()> start1) {
            LLAMA_LOG_INFO("prompt: %s\n", prompt.detokenize(ctx, true, start1, length).c_str());
        }

    }

    void discard_n_kv_and_cache_tokens(llama_context* ctx, server_slot& slot, int32_t n_keep, int32_t n_discard) {
        llama_kv_cache_seq_rm(ctx, slot.id, n_keep, n_keep + n_discard);
        llama_kv_cache_seq_add(ctx, slot.id, n_keep + n_discard, system_tokens.size() + slot.n_past, -n_discard);
        if (slot.params.cache_prompt) {
            slot.cache_tokens.discard_n_tokens(n_keep, n_discard);
        }
    }

    // convert keep first few and discard next tokens in a to b
    void context_shift_find_n_tokens(llama_context* ctx, const server_tokens& a, const server_tokens& b, int32_t n_keep,
        int32_t n_discard, int32_t& n_kept, int32_t& n_discarded, bool exact = false) {
        //n_discarded = n_discard;
        //n_kept = n_keep;

        common_prefix ctx_keep_prefix = a.get_common_prefix_first_n(ctx, b, n_keep, exact);
        common_prefix ctx_total_discard_prefix = a.get_common_prefix_first_n(ctx, b, n_discard + n_keep, exact);
        // only if there is enough common token
        int32_t discard_offset = ctx_total_discard_prefix.first - (n_discard + n_keep);
        int32_t keep_offset = ctx_keep_prefix.first - n_keep;
        //if (ctx_keep_prefix.first == n_keep && ctx_total_discard_prefix.first == n_discard + n_keep)
        //{
        n_kept = ctx_keep_prefix.second - keep_offset;
        n_discarded = ctx_total_discard_prefix.second - ctx_keep_prefix.second - discard_offset;
        if (n_kept < 0) {
            n_kept = n_keep;
        }
        if (n_discarded < 0) {
            n_discarded = n_discard;
        }
        //}
    }

    void context_shift_prompt(llama_context* ctx, server_slot& slot, bool exact = false) {
        //server_tokens prompt_tokens = std::move(slot.prompt_tokens);
        int n_keep = std::max(0, slot.params.n_keep + add_bos_token);
        const int n_left = slot.n_ctx - n_keep;
        int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

        int n_discard_prompt = 0; 
        // we still need to truncate input since we have not discarded enough tokens
        while (slot.n_prompt_tokens - slot.n_discarded_prompt >= slot.n_ctx) {
            slot.n_discarded_prompt = slot.n_discarded_prompt + n_discard;
            n_discard_prompt = n_discard_prompt + n_discard;
        }

        // Handle mistokenization between prompt and cache during context shift
        //
        int32_t n_discard_cache = n_discard_prompt;
        int32_t n_kept = n_keep;
        slot.prompt_tokens.discard_n_tokens(n_keep, slot.n_discarded_prompt - n_discard_prompt);
        if (n_discard_prompt > 0) {
            context_shift_find_n_tokens(ctx, slot.prompt_tokens, slot.cache_tokens, n_keep,
                 n_discard,  n_kept, n_discard_cache, exact);
            //common_prefix ctx_keep_prefix = slot.prompt_tokens.get_common_prefix_first_n(ctx, slot.cache_tokens, n_keep, false);
            //common_prefix ctx_total_discard_prefix = slot.prompt_tokens.get_common_prefix_first_n(ctx, slot.cache_tokens, n_discard_prompt + n_keep, false);

            //if (ctx_keep_prefix.first == n_keep && ctx_total_discard_prefix.first == n_discard_prompt + n_keep) {
            //    n_kept = ctx_keep_prefix.second;
            //    n_discard_cache = ctx_total_discard_prefix.second - ctx_keep_prefix.second;
            //}
        }

        int n_discard_cache_max = std::max((int32_t)slot.cache_tokens.size() - n_kept, 0);
        n_discard_cache = std::min(n_discard_cache, n_discard_cache_max);
        // discard matching tokens from cache and kv cache to avoid reprocessing the prompt
        if (n_discard_cache > 0) {
            discard_n_kv_and_cache_tokens(ctx, slot, n_kept, n_discard_cache);
        }
        // discard extra tokens from prompts
        slot.n_kept_prompt = n_keep;
        slot.prompt_tokens.discard_n_tokens(n_keep, n_discard_prompt);
        slot.n_prompt_tokens = slot.prompt_tokens.size();
    }

    void update_slots() {
        if (system_need_update) {
            system_prompt_update();
        }

        // release slots
        for (auto & slot : slots) {
            if (slot.command == SLOT_COMMAND_RELEASE) {
                slot.state       = SLOT_STATE_IDLE;
                slot.command     = SLOT_COMMAND_NONE;
                slot.t_last_used = ggml_time_us();

                LOG_INFO("slot released", {
                    {"id_slot",         slot.id},
                    {"id_task",         slot.id_task},
                    {"n_ctx",           n_ctx},
                    {"n_past",          slot.n_past},
                    {"n_system_tokens", system_tokens.size()},
                    {"n_cache_tokens",  slot.cache_tokens.size()},
                    {"truncated",       slot.truncated}
                });

                queue_tasks.notify_slot_changed();
            }
        }

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_IDLE || slot.command != SLOT_COMMAND_NONE) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                LOG_INFO("all slots are idle", {});
                if (system_prompt.empty() && clean_kv_cache) {
                    kv_cache_clear();
                }

                return;
            }
        }

        {
            LOG_VERBOSE("posting NEXT_RESPONSE", {});

            server_task task;
            task.type      = SERVER_TASK_TYPE_NEXT_RESPONSE;
            task.id_target = -1;

            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.ga_n == 1) {
                if (slot.is_processing() && (int) system_tokens.size() + slot.n_past >= slot.n_ctx - 1) {
                    if (!params.ctx_shift) {
                        // this check is redundant (for good)
                        // we should never get here, because generation should already stopped in process_token()
                        send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                        slot.release();
                        continue;
                    }
                    if (mctx) {
                        // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                        // we don't support ctx_shift because an image chunk may contains multiple tokens
                        GGML_ABORT("not supported by multimodal");
                    }
                    // Shift context
                    int n_keep = slot.params.n_keep < 0 ? slot.prompt_tokens.size() : slot.params.n_keep;
                    if (add_bos_token) {
                        n_keep += 1;
                    }
                    n_keep = std::min(slot.n_ctx - 4, n_keep);

                    const int n_left    = (int) system_tokens.size() + slot.n_past - n_keep;
                    const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);
                    int32_t n_kept;
                    int32_t n_discard_cache;
                    context_shift_find_n_tokens(ctx, slot.prompt_tokens, slot.cache_tokens, n_keep,
                        n_discard, n_kept, n_discard_cache);

                    LOG_INFO("slot context shift", {
                        {"id_slot",         slot.id},
                        {"id_task",         slot.id_task},
                        {"n_keep",          n_keep},
                        {"n_left",          n_left},
                        {"n_discard",       n_discard},
                        {"n_ctx",           n_ctx},
                        {"n_past",          slot.n_past},
                        {"n_system_tokens", system_tokens.size()},
                        {"n_cache_tokens",  slot.cache_tokens.size()}
                    });
                    slot.n_discarded_prompt = slot.n_discarded_prompt + n_discard;
                    slot.n_kept_prompt = n_keep;
                    discard_n_kv_and_cache_tokens(ctx, slot, n_kept, n_discard_cache);
                    slot.n_past -= n_discard_cache;
                    slot.truncated = true;
                }
            }
        }

        // start populating the batch for this iteration
        llama_batch_clear(batch);

        auto accept_special_token = [&](server_slot& slot, llama_token token) {
            return params.special || slot.sparams.preserved_tokens.find(token) != slot.sparams.preserved_tokens.end();
        };

        // frist, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_IDLE) {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            const int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

            // TODO: we always have to take into account the "system_tokens"
            //       this is not great and needs to be improved somehow
            llama_batch_add(batch, slot.sampled, system_tokens.size() + slot.cache_tokens.pos_next(), { slot.id }, true);

            slot.n_past += 1;

            if (slot.params.cache_prompt) {
                slot.cache_tokens.push_back(slot.sampled);
            }

            LOG_VERBOSE("slot decode token", {
                {"id_slot",         slot.id},
                {"id_task",         slot.id_task},
                {"n_ctx",           n_ctx},
                {"n_past",          slot.n_past},
                {"n_system_tokens", system_tokens.size()},
                {"n_cache_tokens",  slot.cache_tokens.size()},
                {"truncated",       slot.truncated}
            });
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        // track if this is an embedding or non-embedding batch
        // if we've added sampled tokens above, we are in non-embedding mode
        // -1: none, 0: non-embedding, 1: embedding
        int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

        // next, batch any pending prompts without exceeding n_batch
        if (params.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_IDLE && slot.command == SLOT_COMMAND_LOAD_PROMPT) {
                    auto & prompt_tokens = slot.prompt_tokens;

                    // we haven't tokenized the prompt yet - do it now:
                    if (prompt_tokens.empty() || slot.n_prompt_tokens==0 ) {
                        LOG_VERBOSE("tokenizing prompt", {
                            {"id_slot", slot.id},
                            {"id_task", slot.id_task}
                        });

                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        if (slot.infill) {
                            const bool add_bos = llama_should_add_bos_token(model);
                            bool suff_rm_leading_spc = true;
                            if (params.input_suffix.find_first_of(' ') == 0 && params.input_suffix.size() > 1) {
                                params.input_suffix.erase(0, 1);
                                suff_rm_leading_spc = false;
                            }

                            auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                            auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                            const int space_token = 29871; // TODO: this should not be hardcoded
                            if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token) {
                                suffix_tokens.erase(suffix_tokens.begin());
                            }

                            prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
                            suffix_tokens.insert(suffix_tokens.begin(), llama_token_suffix(model));

                            auto embd_inp = params.spm_infill ? suffix_tokens : prefix_tokens;
                            auto embd_end = params.spm_infill ? prefix_tokens : suffix_tokens;
                            if (add_bos) {
                                embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
                            }
                            embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

                            const llama_token middle_token = llama_token_middle(model);
                            if (middle_token >= 0) {
                                embd_inp.push_back(middle_token);
                            }

                            prompt_tokens = server_tokens(embd_inp, false);
                        } else {
                            // prompt_tokens = tokenize(slot.prompt, system_prompt.empty()); // add BOS if there isn't system prompt
                        }

                        slot.n_past = 0;
                        slot.n_prompt_tokens = prompt_tokens.size();

                        LOG_VERBOSE("prompt tokenized", {
                            {"id_slot",         slot.id},
                            {"id_task",         slot.id_task},
                            {"n_ctx",           slot.n_ctx},
                            {"n_keep",          slot.params.n_keep},
                            {"n_prompt_tokens", slot.n_prompt_tokens},
                            {"prompt_tokens", prompt_tokens.detokenize(ctx, true)},
                        });

                        // empty prompt passed -> release the slot and send empty response
                        if (prompt_tokens.empty()) {
                            LOG_INFO("empty prompt - releasing slot", {
                                {"id_slot", slot.id},
                                {"id_task", slot.id_task}
                            });

                            slot.state = SLOT_STATE_PROCESSING;
                            slot.command = SLOT_COMMAND_NONE;
                            slot.release();
                            slot.print_timings();
                            send_final_response(slot);
                            continue;
                        }

                        if (slot.embedding) {
                            // this prompt is too large to process - discard it
                            if (slot.n_prompt_tokens > n_ubatch) {
                                slot.state = SLOT_STATE_PROCESSING;
                                slot.command = SLOT_COMMAND_NONE;
                                slot.release();
                                send_error(slot, "input is too large to process. increase the physical batch size", ERROR_TYPE_SERVER);
                                continue;
                            }
                        } else {
                            // if input prompt is too big, truncate it (if group attention self-extend is disabled)
                            // context shift for prompt processing
                            if (slot.ga_n == 1 && slot.n_prompt_tokens >= slot.n_ctx) {
                                if (!params.ctx_shift) {
                                    send_error(slot, "the request exceeds the available context size, try increasing it", ERROR_TYPE_SERVER);
                                    slot.release();
                                    continue;
                                }
                                if (mctx) {
                                    // we should never reach this because params.ctx_shift is automatically disabled if mmproj is loaded
                                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                                    GGML_ABORT("not supported by multimodal");
                                }

                                context_shift_prompt(ctx, slot);
                                slot.truncated = true;
                                LOG_VERBOSE("input truncated", {
                                    {"id_slot",         slot.id},
                                    {"id_task",         slot.id_task},
                                    {"n_ctx",           slot.n_ctx},
                                    {"n_keep",          slot.params.n_keep},
                                    {"n_left",          slot.n_ctx- slot.params.n_keep},
                                    {"n_prompt_tokens", slot.n_prompt_tokens},
                                    {"prompt_tokens",   prompt_tokens.detokenize(ctx, true)},
                                    });

                                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                                
#ifndef NDEBUG
                                // debug
                                common_prefix prefix = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, false);
                                int32_t back = 1;
                                if (slot.cache_tokens.size() && slot.cache_tokens.size() > prefix.first+20
                                    && prefix.second >= back && prefix.first >= back) {
                                    LLAMA_LOG_INFO("After context shift :\n");
                                    print_tokens(slot.prompt_tokens, slot.cache_tokens, prefix.second - back, prefix.first - back, 50);
                                }
#endif 
                            }
                            else {
                                slot.n_discarded_prompt = 0;
                            }
                            llama_sampling_reset(llama_get_model_vocab(model), slot.ctx_sampling);

                            if (!slot.params.cache_prompt) {
                                slot.n_past_se = 0;
                                slot.ga_i      = 0;
                            } else {
                                GGML_ASSERT(slot.ga_n == 1);

                                // reuse any previously computed tokens that are common with the new prompt
                                common_prefix prefix = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, true);
                                common_prefix prefix_nonexact = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, false);
                                auto n_past0 = slot.cache_tokens.get_common_prefix_exact(prompt_tokens);
                                LLAMA_LOG_INFO("Cache: cache_size = %ld, n_past0 =  %ld, n_past1 =  %ld, n_past_prompt1 = %ld,  n_past2 =  %ld, n_past_prompt2 =  %ld\n", (int32_t) slot.cache_tokens.size(), (int32_t) n_past0, (int32_t) prefix.first, prefix.second, (int32_t) prefix_nonexact.first, (int32_t) prefix_nonexact.second);
                                if (prefix.first + 20 < prefix_nonexact.first) {
                                    LLAMA_LOG_WARN("Common part contains missing or extra space and new line\n");
                                    prefix = prefix_nonexact;
                                }
                                slot.n_past = prefix.first;
                                slot.n_past_prompt = prefix.second;
                                GGML_ASSERT(slot.n_past <= slot.cache_tokens.size() && "n_past cannot be larger than size");

                                bool cache_equal = prompt_cache_equal(ctx, slot.cache_tokens,
                                    slot.prompt_tokens,  0, prefix);
                                if (slot.n_past != slot.n_past_prompt) {
                                    LLAMA_LOG_INFO("Mistokenization found and handled successfully.\n");
                                }
                                if ((slot.n_past + 20 <=slot.cache_tokens.size() || !cache_equal))
                                {
                                    LLAMA_LOG_WARN("Common part does not match fully\n");
                                    int32_t back = 4;
                                    if (prefix.second >= back && prefix.first >= back) {
                                        print_tokens(slot.prompt_tokens, slot.cache_tokens, prefix.second - back, prefix.first - back, 30);
                                    }
                                }

                                // push the prompt into the sampling context (do not apply grammar)
                                for (int i = 0; i < slot.n_past; ++i) {
                                    llama_sampling_accept(slot.ctx_sampling, ctx, slot.cache_tokens[i], false);
                                }
                            }
                        }

                        if (slot.n_past_prompt == slot.n_prompt_tokens && slot.n_past_prompt > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            LOG_INFO("we have to evaluate at least 1 token to generate logits", {
                                { "id_slot", slot.id },
                                { "id_task", slot.id_task }
                            });

                            slot.n_past_prompt--;
                            slot.n_past--;
                            if (slot.ga_i > 0) {
                                slot.n_past_se--;
                            }
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    if (slot.embedding) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
                            continue;
                        }
                    }

                    // check that we are in the right batch_type, if not defer the slot
                    bool slot_type = slot.embedding ? 1 : 0;
                    if (batch_type == -1) {
                        batch_type = slot_type;
                    } else if (batch_type != slot_type) {
                        continue;
                    }

                    // keep only the common part
                    // remove the non-common part from the cache
                    slot.cache_tokens.keep_first(slot.n_past);
                    int p0 = (int) system_tokens.size() + slot.n_past;
                    p0 = system_tokens.size() + slot.cache_tokens.pos_next();
                    if (!llama_kv_cache_seq_rm(ctx, slot.id, p0, -1)) {
                        // could not partially delete (likely using a non-Transformer model)
                        llama_kv_cache_seq_rm(ctx, slot.id, -1, -1);

                        p0 = (int) system_tokens.size();
                        if (p0 != 0) {
                            // copy over the system prompt when there is one
                            llama_kv_cache_seq_cp(ctx, 0, slot.id, -1, -1);
                        }

                        // there is no common part left (except for the system prompt)
                        slot.n_past = 0;
                        slot.n_past_se = 0;
                        slot.ga_i = 0;
                        // TODO: is the system prompt ever in the sampling context?
                        llama_sampling_reset(llama_get_model_vocab(model), slot.ctx_sampling);
                    }

                    LOG_INFO("kv cache rm [p0, end)", {
                        { "id_slot", slot.id },
                        { "id_task", slot.id_task },
                        { "p0",      p0 }
                    });

                    // check if we should process the image
                    if (slot.n_past_prompt < slot.n_prompt_tokens
                        && slot.prompt_tokens[slot.n_past_prompt] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        llama_pos p1 = slot.cache_tokens.pos_next()+slot.n_past_prompt-slot.n_past; // add offset to prompt
                        int32_t res = slot.prompt_tokens.process_chunk(ctx, mctx, slot.n_past_prompt, p1, slot.id, n_tokens_out);
                        if (res != 0) {
                            LLAMA_LOG_ERROR("failed to process image, res = %d\n", res);
                            slot.release();
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            continue;
                        }

                        // add the image chunk to cache
                        {
                            const auto& chunk = slot.prompt_tokens.find_chunk(slot.n_past_prompt);
                            slot.cache_tokens.push_back(chunk.get()); // copy
                        }

                        slot.n_past += n_tokens_out;
                        slot.n_past_prompt += n_tokens_out;
                        slot.n_prompt_tokens_processed += n_tokens_out;

                    }



                    int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

                    int32_t ga_i = slot.ga_i;
                    int32_t ga_n = slot.ga_n;
                    int32_t ga_w = slot.ga_w;

                    // add prompt tokens for processing in the current batch
                    // TODO: the self-extend stuff here is a mess - simplify and/or abstract it somehow
                    while (slot.n_past_prompt < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = slot.prompt_tokens[slot.n_past_prompt];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }
                        if (slot.ga_n != 1) {
                            while (slot_npast >= ga_i + ga_w) {
                                const int bd = (ga_w/ga_n)*(ga_n - 1);
                                slot_npast -= bd;
                                ga_i += ga_w/ga_n;
                            }
                        }

						int p0=system_tokens.size() + slot.cache_tokens.pos_next();
                        llama_batch_add(batch, cur_tok, p0, { slot.id }, false);
                        
                        slot.cache_tokens.push_back(cur_tok);
                        

                        slot.n_prompt_tokens_processed++;
                        slot_npast++;
                        slot.n_past_prompt++;
                        slot.n_past++;
                    }
                    LOG_VERBOSE("prompt processing progress", {
                        {"id_slot",  slot.id},
                        {"n_past",   slot.n_past},
                        {"n_ctx",    n_ctx},
                        {"n_tokens", batch.n_tokens},
                        {"progress", (float) slot.n_prompt_tokens_processed / slot.n_prompt_tokens},
                    });

                    // entire prompt has been processed - start decoding new tokens
                    if (slot.n_past_prompt == slot.n_prompt_tokens) {
                        slot.state   = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;

                        GGML_ASSERT(batch.n_tokens > 0);
                        GGML_ASSERT((size_t)slot.n_prompt_tokens == slot.prompt_tokens.size());
                        llama_sampling_reset(llama_get_model_vocab(model), slot.ctx_sampling);
                        for (int i = 0; i < slot.n_prompt_tokens; ++i) {
                            llama_token id = slot.prompt_tokens[i];
                            if (id != LLAMA_TOKEN_NULL) {
                                llama_sampling_accept(slot.ctx_sampling, ctx, id, false);
                            }
                        }

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        LOG_VERBOSE("prompt done", {
                            {"id_slot",  slot.id},
                            {"n_past",   slot.n_past},
                            {"n_ctx",    n_ctx},
                            {"n_tokens", batch.n_tokens},
                        });
                    }
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        if (batch.n_tokens == 0) {
            LOG_VERBOSE("no tokens to decode", {});
            return;
        }

        LOG_VERBOSE("decoding batch", {
            {"n_tokens", batch.n_tokens},
        });

        // make sure we're in the right embedding mode
        llama_set_embeddings(ctx, batch_type == 1);

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            for (auto & slot : slots) {
                if (slot.ga_n != 1) {
                    // context extension via Self-Extend
                    // TODO: simplify and/or abstract this
                    while (slot.n_past_se >= slot.ga_i + slot.ga_w) {
                        const int ib = (slot.ga_n * slot.ga_i) / slot.ga_w;
                        const int bd = (slot.ga_w / slot.ga_n) * (slot.ga_n - 1);
                        const int dd = (slot.ga_w / slot.ga_n) - ib * bd - slot.ga_w;

                        LOG_TEE("\n");
                        LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i, slot.n_past_se, ib * bd, slot.ga_i + ib * bd, slot.n_past_se + ib * bd);
                        LOG_TEE("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n, (slot.ga_i + ib * bd) / slot.ga_n, (slot.ga_i + ib * bd + slot.ga_w) / slot.ga_n);
                        LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd, slot.ga_i + ib * bd + slot.ga_w + dd, slot.n_past_se + ib * bd + dd);

                        llama_kv_cache_seq_add(ctx, slot.id, slot.ga_i, slot.n_past_se, ib * bd);
                        llama_kv_cache_seq_div(ctx, slot.id, slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n);
                        llama_kv_cache_seq_add(ctx, slot.id, slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd);

                        slot.n_past_se -= bd;

                        slot.ga_i += slot.ga_w / slot.ga_n;

                        LOG_TEE("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", slot.n_past_se + bd, slot.n_past_se, slot.ga_i);
                    }

                    slot.n_past_se += n_tokens;
                }
            }

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);

            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_ERROR("failed to decode the batch: KV cache is full - try increasing it via the context size", {
                        {"i",   i},
                        {"n_batch",  ret},
                        {"ret",   ret},
                    });
                    for (auto & slot : slots) {
                        slot.state = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;
                        slot.release();
                        LLAMA_LOG_INFO("n_past =% d\n", slot.cache_tokens.size());
                        send_error(slot, "Input prompt is too big compared to KV size. Please try increasing KV size.");
                    }
                    break; // break loop of n_batch
                }


                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                LOG_WARNING("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation", {
                    {"i",   i},
                    {"n_batch",  n_batch},
                    {"ret",   ret},
                });

                continue; // continue loop of n_batch
            }

            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                // prompt evaluated for embedding
                if (slot.embedding) {
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    continue; // continue loop of slots
                }

                completion_token_output result;
                const int tok_idx = slot.i_batch - i;
                const llama_token id = llama_sampling_sample(slot.ctx_sampling, ctx, NULL, tok_idx);

                llama_sampling_accept(slot.ctx_sampling, ctx, id, true);

                slot.n_decoded += 1;

                const int64_t t_current = ggml_time_us();

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = (t_current - slot.t_start_generation) / 1e3;

                result.tok = id;
                result.prob = 1.0f; // TODO: set it here instead of doing inside populate_token_probs
                result.text_to_send = llama_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));

                if (slot.sparams.n_probs > 0) {
                    populate_token_probs(slot, result, slot.params.post_sampling_probs, params.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                }

                slot.i_batch = -1;
            }

            // Do speculative decoding
            for (auto & slot : slots) {
                if (!slot.is_processing() || !slot.spec) {
                    continue;
                }

                if (slot.state != SLOT_STATE_PROCESSING) {
                    continue;
                }

                if (mctx) {
                    // we should never reach this, as speculative is automatically disabled if mmproj is loaded
                    GGML_ABORT("not supported by multimodal");
                }

                // determine the max draft that fits the current slot state
                int n_draft_max = slot.params.speculative.n_max;

                // note: n_past is not yet increased for the `id` token sampled above
                //       also, need to leave space for 1 extra token to allow context shifts
                n_draft_max = std::min(n_draft_max, slot.n_ctx - slot.n_past - 2);

                if (slot.n_predict > 0) {
                    n_draft_max = std::min(n_draft_max, slot.n_predict - slot.n_decoded - 1);
                }

                LOG_VERBOSE("max possible draft", {
                    {"id_slot", slot.id},
                    {"n_draft_max", n_draft_max}
                });

                if (n_draft_max < slot.params.speculative.n_min) {
                    LOG_VERBOSE("the max possible draft is too small", {
                        {"id_slot", slot.id},
                        {"n_draft_max", n_draft_max},
                        {"n_min", slot.params.speculative.n_min}
                    });
                    continue;
                }

                llama_token id = slot.sampled;

                struct llama_speculative_params params_spec;
                params_spec.n_draft = n_draft_max;
                params_spec.n_reuse = cparams_dft.n_ctx - slot.params.speculative.n_max;
                params_spec.p_min = slot.params.speculative.p_min;

                const std::vector<llama_token> & cached_text_tokens = slot.cache_tokens.tokens_data();
                std::vector<llama_token> draft = llama_speculative_gen_draft(slot.spec, params_spec, cached_text_tokens, id);

                // ignore small drafts
                if (slot.params.speculative.n_min > (int) draft.size()) {
                    LOG_VERBOSE("ignoring small draft", {
                        {"id_slot", slot.id},
                        {"draft_size", (int) draft.size()},
                        {"n_min", slot.params.speculative.n_min}
                    });
                    continue;
                }

                // keep track of total number of drafted tokens tested
                slot.n_draft_total += draft.size();

                // construct the speculation batch
                llama_batch_clear(slot.batch_spec);
                llama_batch_add(slot.batch_spec, id, slot.cache_tokens.pos_next(), { slot.id }, true);

                for (size_t i = 0; i < draft.size(); ++i) {
                    llama_batch_add(slot.batch_spec, draft[i], slot.cache_tokens.pos_next() + 1 + i, { slot.id }, true);
                }

                LOG_VERBOSE("decoding speculative batch", {
                    {"id_slot", slot.id},
                    {"size", slot.batch_spec.n_tokens}
                });

                llama_decode(ctx, slot.batch_spec);

                // the accepted tokens from the speculation
                std::vector<llama_token> ids = llama_sampling_sample_and_accept_n(slot.ctx_sampling, ctx, draft);

                slot.n_past += ids.size();
                slot.n_decoded += ids.size();

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                slot.cache_tokens.push_back(id);
                slot.cache_tokens.insert({ ids.begin(), ids.end() - 1 });

                llama_kv_cache_seq_rm(ctx, slot.id, slot.n_past, -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok = ids[i];
                    result.text_to_send = llama_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    if (slot.sparams.n_probs > 0) {
                        populate_token_probs(slot, result, slot.params.post_sampling_probs, params.special, i);
                    }

                    if (!process_token(result, slot)) {
                        // release slot because of stop condition
                        slot.release();
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        break;
                    }
                }

                LOG_VERBOSE("speculative decoding result", {
                    {"id_slot", slot.id},
                    {"accepted", (int) ids.size() - 1},
                    {"total", (int) draft.size()},
                    {"new_n_past", slot.n_past}
                });
            }
        }

        LOG_VERBOSE("run slots completed", {});
    }

    json model_meta() const {
        return json {
            {"vocab_type",  llama_vocab_type    (model)},
            {"n_vocab",     llama_n_vocab       (model)},
            {"n_ctx_train", llama_n_ctx_train   (model)},
            {"n_embd",      llama_n_embd        (model)},
            {"n_params",    llama_model_n_params(model)},
            {"size",        llama_model_size    (model)},
        };
    }
};

static json format_final_response_oaicompat(const json& request, json result, const std::string& completion_id, bool streaming = false) {
    bool stopped_word = result.count("stopped_word") != 0;
    bool stopped_eos = json_value(result, "stopped_eos", false);
    int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
    int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason = "length";
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json choices =
        streaming ? json::array({ json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}} })
        : json::array({ json{{"finish_reason", finish_reason},
                              {"index", 0},
                              {"message", json{{"content", content},
                                               {"role", "assistant"}}}} });

    std::time_t t = std::time(0);

    json res = json{
        {"choices", choices},
        {"created", t},
        {"model",
            json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", streaming ? "chat.completion.chunk" : "chat.completion"},
        {"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }},
        {"id", completion_id}
    };

    if (server_verbose) {
        res["__verbose"] = result;
    }

    if (result.contains("completion_probabilities")) {
        res["completion_probabilities"] = json_value(result, "completion_probabilities", json::array());
    }

    return res;
}

// return value is vector as there is one case where we might need to generate two responses
static std::vector<json> format_partial_response_oaicompat(server_task_result task_result, const std::string& completion_id) {
    json result = task_result.data;
    std::cout << result.dump(4) << std::endl;
    if (!result.contains("model") || !result.contains("oaicompat_token_ctr")) {
        return std::vector<json>({ result });
    }

    bool first = json_value(result, "oaicompat_token_ctr", 0) == 0;
    std::string modelname = json_value(result, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    bool stopped_word = json_value(result, "stopped_word", false);
    bool stopped_eos = json_value(result, "stopped_eos", false);
    bool stopped_limit = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason;
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    std::time_t t = std::time(0);

    json choices;

    if (!finish_reason.empty()) {
        choices = json::array({ json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}} });
    }
    else {
        if (first) {
            if (content.empty()) {
                choices = json::array({ json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}} });
            }
            else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{ {"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"} };

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"} };

                return std::vector<json>({ initial_ret, second_ret });
            }
        }
        else {
            // Some idiosyncrasy in task processing logic makes several trailing calls
            // with empty content, we ignore these at the calee site.
            if (content.empty()) {
                return std::vector<json>({ json::object() });
            }

            choices = json::array({ json{
                {"finish_reason", nullptr},
                {"index", 0},
                {"delta",
                json{
                    {"content", content},
                }},
            } });
        }
    }

    json ret = json{
        {"choices", choices},
        {"created", t},
        {"id",      completion_id},
        {"model",   modelname},
        {"object",  "chat.completion.chunk"}
    };

    if (task_result.timings.prompt_n != -1) {
        ret.push_back({ "timings", task_result.timings.to_json() });
    }

    //
    if (!finish_reason.empty()) {
        int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
        int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
        ret.push_back({ "usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        } });
    }

    return std::vector<json>({ ret });
}


//static json format_embeddings_response_oaicompat(const json& request, const json& embeddings) {
//    json data = json::array();
//    int32_t n_tokens = 0;
//    int i = 0;
//    for (auto& elem : embeddings) {
//        data.push_back(json{
//            {"embedding", json_value(elem, "embedding", json::array())},
//            {"index",     i++},
//            {"object",    "embedding"}
//            });
//    }
//
//    json res = json{
//        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
//        {"object", "list"},
//        {"usage", json {
//            {"prompt_tokens", n_tokens},
//            {"total_tokens", n_tokens}
//        }},
//        {"data", data}
//    };
//
//    return res;
//}

static json format_embeddings_response_oaicompat(const json& request, const json& embeddings, bool use_base64 = false) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto& elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        }
        else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);
        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }
    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    LOG_INFO("request", {
        {"remote_addr", req.remote_addr},
        {"remote_port", req.remote_port},
        {"status",      res.status},
        {"method",      req.method},
        {"path",        req.path},
        {"params",      req.params},
    });

    LOG_VERBOSE("request", {
        {"request",  req.body},
        {"response", res.body},
    });
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char ** argv) {
#if SERVER_VERBOSE != 1
    log_disable();
#endif
    // own arguments required by this example
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // parse arguments from environment variables
    gpt_params_parse_from_env(params);

    // TODO: not great to use extern vars
    server_log_json = params.log_json;
    server_verbose = params.verbosity > 0;


    // struct that contains llama context and inference
    server_context ctx_server;

    if (!params.system_prompt.empty()) {
        ctx_server.system_prompt_set(params.system_prompt);
    }

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INFO("build info", {
        {"build",  LLAMA_BUILD_NUMBER},
        {"commit", LLAMA_COMMIT}
    });

    LOG_INFO("system info", {
        {"n_threads",       params.n_threads},
        {"n_threads_batch", params.n_threads_batch},
        {"total_threads",   std::thread::hardware_concurrency()},
        {"system_info",     llama_print_system_info()},
    });

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INFO("Running with SSL", {{"key", params.ssl_file_key}, {"cert", params.ssl_file_cert}});
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INFO("Running without SSL", {});
        svr.reset(new httplib::Server());
    }
#else
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "ik_llama.cpp"}});

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, json error_data) {
        json final_response {{"error", error_data}};
        res.set_content(final_response.dump(), "application/json; charset=utf-8");
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok = [](httplib::Response& res, const json& data) {
        res.set_content(data.dump(), "application/json; charset=utf-8");
        res.status = 200;
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message;
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_VERBOSE("Got exception", formatted_error);
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    if (!svr->bind_to_port(params.hostname, params.port)) {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", params.hostname.c_str(), params.port);
        return 1;
    }

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;
    ctx_server.cache_ram_n_min = params.cache_ram_n_min;
    ctx_server.cache_ram_similarity = params.cache_ram_similarity;
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    auto db_handle = std::make_shared<DatabaseHandle>(params.sql_save_file);
    bool sqlite_extension_loaded = false;
    if (!params.sqlite_zstd_ext_file.empty()) {
        auto* conn = db_handle->db.connection().get();
        sqlite3_enable_load_extension(conn, 1);
        char* errmsg = nullptr;
        const int rc = sqlite3_load_extension(
            conn,
            params.sqlite_zstd_ext_file.c_str(),
            nullptr,
            &errmsg
        );
        if(rc != SQLITE_OK) {
            const std::string err = errmsg ? errmsg : "Unknown extension error";
            sqlite3_free(errmsg);
            LOG_WARNING("Failed to load extension", {{"err", err}});
        }
	else {
            sqlite_extension_loaded = true;
        }
        sqlite3_enable_load_extension(conn, 0);
    }
#else
    auto db_handle = false;
#endif
    // load the model
    if (!ctx_server.load_model(params)) {
        state.store(SERVER_STATE_ERROR);
        return 1;
    } else {
        ctx_server.init();
        state.store(SERVER_STATE_READY);
    }

    LOG_INFO("model loaded", {});

    const auto model_meta = ctx_server.model_meta();

    // print sample chat example to make it clear which template is used

        LOG_INFO("chat template", {
        {"chat_template", common_chat_templates_source(ctx_server.chat_templates.get())},
    });

    LOG_INFO("chat template", {
        {"chat_example", common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params.use_jinja, {}).c_str()
        },
            {"built_in",     params.chat_template.empty()},
        });
    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params, &res_error](const httplib::Request & req, httplib::Response & res) {
        // TODO: should we apply API key to all endpoints, including "/health" and "/models"?
        static const std::set<std::string> protected_endpoints = {
            "/props",
            "/completion",
            "/completions",
            "/v1/completions",
            "/chat/completions",
            "/v1/chat/completions",
            "/infill",
            "/tokenize",
            "/detokenize",
            "/embedding",
            "/embeddings",
            "/v1/embeddings",
        };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is not in protected_endpoints list, skip validation
        if (protected_endpoints.find(req.path) == protected_endpoints.end()) {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true; // API key is valid
            }
        }

        // API key is invalid or not provided
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WARNING("Unauthorized: Invalid API Key", {});

        return false;
    };

    auto middleware_server_state = [&res_error, &state](const httplib::Request& req, httplib::Response& res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            }
            else if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
                // allow the models endpoint to be accessed during loading
                return true;
            }
            else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
        });

    //
    // Route handlers (or controllers)
    //

    const auto handle_health = [&](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        switch (current_state) {
            case SERVER_STATE_READY:
                {
                    // request slots data using task queue
                    server_task task;
                    task.id   = ctx_server.queue_tasks.get_new_id();
                    task.type = SERVER_TASK_TYPE_METRICS;
                    task.id_target = -1;

                    ctx_server.queue_results.add_waiting_task_id(task.id);
                    ctx_server.queue_tasks.post(std::move(task));

                    // get the result
                    server_task_result result = ctx_server.queue_results.recv(task.id);
                    ctx_server.queue_results.remove_waiting_task_id(task.id);

                    const int n_idle_slots       = result.data.at("idle");
                    const int n_processing_slots = result.data.at("processing");

                    json health = {
                        {"status",           "ok"},
                        {"slots_idle",       n_idle_slots},
                        {"slots_processing", n_processing_slots}
                    };

                    res.status = 200; // HTTP OK
                    if (params.endpoint_slots && req.has_param("include_slots")) {
                        health["slots"] = result.data.at("slots");
                    }

                    if (n_idle_slots == 0) {
                        health["status"] = "no slot available";
                        if (req.has_param("fail_on_no_slot")) {
                            res.status = 503; // HTTP Service Unavailable
                        }
                    }

                    res.set_content(health.dump(), "application/json");
                    break;
                }
            case SERVER_STATE_LOADING_MODEL:
                {
                    res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
                } break;
            case SERVER_STATE_ERROR:
                {
                    res_error(res, format_error_response("Model failed to load", ERROR_TYPE_SERVER));
                } break;
        }
    };

    const auto handle_slots = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_slots) {
            res_error(res, format_error_response("This server does not support slots endpoint.", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_multi  = -1;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(std::move(task));

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        res.set_content(result.data.at("slots").dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_metrics) {
            res_error(res, format_error_response("This server does not support metrics endpoint.", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_multi  = -1;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;
        task.data.push_back({{"reset_bucket", true}});

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(std::move(task));

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        json data = result.data;

        const uint64_t n_prompt_tokens_processed = data.at("n_prompt_tokens_processed");
        const uint64_t t_prompt_processing       = data.at("t_prompt_processing");

        const uint64_t n_tokens_predicted  = data.at("n_tokens_predicted");
        const uint64_t t_tokens_generation = data.at("t_tokens_generation");

        const int32_t kv_cache_used_cells = data.at("kv_cache_used_cells");

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) data.at("n_prompt_tokens_processed_total")}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) data.at("t_prompt_processing_total") / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) data.at("n_tokens_predicted_total")}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) data.at("t_tokens_generation_total") / 1.e3}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  n_prompt_tokens_processed ? 1.e3 / t_prompt_processing * n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  n_tokens_predicted ? 1.e3 / t_tokens_generation * n_tokens_predicted : 0.}
            },{
                    {"name",  "kv_cache_usage_ratio"},
                    {"help",  "KV-cache usage. 1 means 100 percent usage."},
                    {"value",  1. * kv_cache_used_cells / params.n_ctx}
            },{
                    {"name",  "kv_cache_tokens"},
                    {"help",  "KV-cache tokens."},
                    {"value",  (uint64_t) data.at("kv_cache_tokens_count")}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of request processing."},
                    {"value",  (uint64_t) data.at("processing")}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of request deferred."},
                    {"value",  (uint64_t) data.at("deferred")}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        const int64_t t_start = data.at("t_start");
        res.set_header("Process-Start-Time-Unix", std::to_string(t_start));

        res.set_content(prometheus.str(), "text/plain; version=0.0.4");
        res.status = 200; // HTTP OK
    };

    const auto handle_slots_save = [&ctx_server, &res_error, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_SAVE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath }
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_restore = [&ctx_server, &res_error, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath }
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_erase = [&ctx_server, &res_error](const httplib::Request & /* req */, httplib::Response & res, int id_slot) {
        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_ERASE;
        task.data = {
            { "id_slot", id_slot },
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_action = [&res_error, &handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request & req, httplib::Response & res) {
        std::string id_slot_str = req.path_params.at("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string action = req.get_param_value("action");

        if (action == "save") {
            handle_slots_save(req, res, id_slot);
        } else if (action == "restore") {
            handle_slots_restore(req, res, id_slot);
        } else if (action == "erase") {
            handle_slots_erase(req, res, id_slot);
        } else {
            res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_props = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        std::string template_key = "tokenizer.chat_template", curr_tmpl;
        int32_t tlen = llama_model_meta_val_str(ctx_server.model, template_key.c_str(), nullptr, 0);
        if (tlen > 0) {
            std::vector<char> curr_tmpl_buf(tlen + 1, 0);
            if (llama_model_meta_val_str(ctx_server.model, template_key.c_str(), curr_tmpl_buf.data(), curr_tmpl_buf.size()) == tlen) {
                curr_tmpl = std::string(curr_tmpl_buf.data(), tlen);
            }
        }
        json data = {
            { "system_prompt",               ctx_server.system_prompt.c_str() },
            { "model_alias",                 ctx_server.params.model_alias },
            { "model_path",                  ctx_server.params.model},
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params.n_parallel },
            { "model_name",                  get_model_name(ctx_server.params.model)},
            { "chat_template",               common_chat_templates_source(ctx_server.chat_templates.get()) },
            { "bos_token",                   llama_token_to_piece(ctx_server.ctx, llama_token_bos(ctx_server.model), /* special= */ true)},
            { "eos_token",                   llama_token_to_piece(ctx_server.ctx, llama_token_eos(ctx_server.model), /* special= */ true)},
            { "model_path",                  ctx_server.params.model },
            { "modalities",                  json {
                {"vision", ctx_server.oai_parser_opt.allow_image},
                {"audio",  ctx_server.oai_parser_opt.allow_audio},
            } },
            { "n_ctx",                       ctx_server.n_ctx }

        };

        if (ctx_server.params.use_jinja) {
            if (auto tool_use_src = common_chat_templates_source(ctx_server.chat_templates.get(), "tool_use")) {
                data["chat_template_tool_use"] = tool_use_src;
        }
        }
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_props_simple = [&ctx_server](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        int n_past = 0;
        int slot_id = 0;
        for (server_slot& slot : ctx_server.slots) {
            if (slot.n_past > n_past) {
                n_past = slot.n_past;
                slot_id = slot.id;
            }
        }
        json data = {
            { "model_name",                  get_model_name(ctx_server.params.model)},
            { "model_path",                  ctx_server.params.model },
            { "modalities",                  json {
                {"vision", ctx_server.oai_parser_opt.allow_image},
                {"audio",  ctx_server.oai_parser_opt.allow_audio},
            } },
             { "n_ctx",                       ctx_server.n_ctx }
        };
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };


    // handle completion-like requests (completion, chat, infill)
    // we can optionally provide a custom format for partial results and final results
    const auto handle_completions_impl = [&ctx_server, &params, &res_error, &res_ok](
        server_task_type type,
        json& data,
        const std::vector<raw_buffer>& files,
        httplib::Response& res,
        oaicompat_type oaicompat) -> void {
            GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION);
            if (ctx_server.params.embedding) {
                res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
                return;
            }

            const auto& prompt = data.at("prompt");

            // process prompt
            std::vector<server_tokens> inputs;

            if (oaicompat && ctx_server.mctx != nullptr) {
                // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
#ifndef NDEBUG
                print_files_info(files);
#endif // !NDEBUG
                inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
            }
            else {
                // Everything else, including multimodal completions.
                inputs = tokenize_input_prompts(llama_get_vocab(ctx_server.ctx), ctx_server.mctx, prompt, true, true);
            }
            const auto completion_id = gen_chatcmplid();
            const int id_task = ctx_server.queue_tasks.get_new_id();

            ctx_server.queue_results.add_waiting_task_id(id_task);
            ctx_server.request_completion(id_task, -1, data, false, false, std::move(inputs[0]));
            bool stream = json_value(data, "stream", false);
            if (!stream) {
                server_task_result result = ctx_server.queue_results.recv(id_task);
                result.oaicompat = oaicompat;
                result.oaicompat_cmpl_id = completion_id;
                json result_oai;
                if (oaicompat) {
                    if (result.final_result) {
                        result_oai = result.to_json_final();
                    }
                    else {
                        result_oai = result.to_json_partial();
                    }
                }
                else {
                    // legacy completions
                    result_oai = result.data;
                }
                if (!result.error && result.stop) {
                    res.set_content(result_oai.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
                }
                else {
                    res_error(res, result_oai);
                }
                ctx_server.queue_results.remove_waiting_task_id(id_task);
            }
            else {
                const auto chunked_content_provider = [id_task, &ctx_server, completion_id, oaicompat, send_done = params.send_done](size_t, httplib::DataSink& sink) {
                    bool successful_completion = false;
                    while (true) {
                        server_task_result result = ctx_server.queue_results.recv(id_task);
                        if (!result.error) {
                            result.oaicompat = oaicompat;
                            result.oaicompat_cmpl_id = completion_id;
                            json res_json;
                            if (oaicompat) {
                                if (result.final_result) {
                                    res_json = result.to_json_final();
                                }
                                else {
                                    res_json = result.to_json_partial();
                                }
                            }
                            else {
                                // legacy completions
                                res_json = result.data;
                            }
                            if (res_json.is_array()) {
                                // chat completions and oai completions
                                for (const auto& res : res_json) {
                                    if (!server_sent_event(sink, res)) {
                                        // sending failed (HTTP connection closed), cancel the generation
                                        ctx_server.queue_results.remove_waiting_task_id(id_task);
                                        return false;
                                    }
                                }
                                if (result.stop) {
                                    successful_completion = true;
                                    break;
                                }
                            }
                            else {
                                // legacy completions
                                if (!server_sent_event(sink, res_json)) {
                                    ctx_server.queue_results.remove_waiting_task_id(id_task);
                                    return false;
                                }
                                if (result.stop) {
                                    break;
                                }
                            }
                        }
                        else {
                            if (!server_sent_event(sink, result.data)) {
                                ctx_server.queue_results.remove_waiting_task_id(id_task);
                                return false;
                            }
                            break;
                        }
                    }
                    bool ok = true;
                    if (successful_completion) {
                        static const std::string done_message = "data: [DONE]\n\n";
                        LOG_VERBOSE("data stream", { {"to_send", done_message} });
                        if (!sink.write(done_message.c_str(), done_message.size())) {
                            // If writing [DONE] fails, the stream is likely already problematic.
                            ok = false;
                        }
                    }
                    sink.done();
                    ctx_server.queue_results.remove_waiting_task_id(id_task);
                    return ok;
                };

                auto on_complete = [id_task, &ctx_server](bool) {
                    // cancel request
                    ctx_server.request_cancel(id_task);
                    ctx_server.queue_results.remove_waiting_task_id(id_task);
                };

                res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
            }
    };

    const auto handle_completions = [&handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        auto data = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            res,
            OAICOMPAT_TYPE_NONE);
    };

    const auto handle_completions_oai = [&handle_completions_impl](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            res,
            OAICOMPAT_TYPE_COMPLETION);
    };

    const auto handle_models = [&params, &model_meta](const httplib::Request & req, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", {
                 {
                     {"id",       params.model_alias},
                     {"object",   "model"},
                     {"created",  std::time(0)},
                     {"owned_by", "llamacpp"},
                     {"meta",     model_meta}
                 },
             }}
        };

        res.set_content(models.dump(), "application/json; charset=utf-8");
    };



    const auto handle_chat_completions = [&ctx_server, &params, &handle_completions_impl, &res_error](const httplib::Request & req, httplib::Response & res) {
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(ctx_server.model, body, ctx_server.oai_parser_opt, files);
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            res,
            OAICOMPAT_TYPE_CHAT);
    };

    // same with handle_chat_completions, but without inference part
    const auto handle_apply_template = [&ctx_server, &params, &res_ok](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy, unused
        json data = oaicompat_chat_params_parse(ctx_server.model, body,ctx_server.oai_parser_opt, files);
        res_ok(res, { { "prompt", std::move(data.at("prompt")) } });
    };

    const auto handle_infill = [&ctx_server, &res_error, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        const int id_task = ctx_server.queue_tasks.get_new_id();
        server_tokens token; // dummy tokens
        ctx_server.queue_results.add_waiting_task_id(id_task);
        ctx_server.request_completion(id_task, -1, data, true, false, std::move(token));
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            res,
            OAICOMPAT_TYPE_NONE); // infill is not OAI compatible
    };

    const auto handle_tokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::vector<llama_token> tokens;
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            tokens = ctx_server.tokenize(body.at("content"), add_special);
        }
        const json data = format_tokenizer_response(tokens);
        return res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_detokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const std::vector<llama_token> tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        return res.set_content(data.dump(), "application/json; charset=utf-8");
    };


    const auto handle_embeddings = [&ctx_server, &res_error](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);
        bool is_openai = false;

        // an input prompt can be a string or a list of tokens (integer)
        json prompt;
        if (body.count("input") != 0) {
            is_openai = true;
            prompt = body.at("input");
        } else if (body.count("content") != 0) {
            // with "content", we only support single prompt
            prompt = std::vector<std::string>{body.at("content")};
        } else {
            res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        // create and queue the task
        json responses;
        {
            const int id_task = ctx_server.queue_tasks.get_new_id();
            ctx_server.queue_results.add_waiting_task_id(id_task);
            std::vector<server_tokens> inputs;
            inputs = tokenize_input_prompts(llama_get_vocab(ctx_server.ctx), ctx_server.mctx, prompt, true, true);
            ctx_server.request_completion(id_task, -1, {{"prompt", prompt}}, false, true, std::move(inputs[0]));

            // get the result
            server_task_result result = ctx_server.queue_results.recv(id_task);
            ctx_server.queue_results.remove_waiting_task_id(id_task);
            if (!result.error) {
                if (result.data.count("results")) {
                    // result for multi-task
                    responses = result.data.at("results");
                } else {
                    // result for single task
                    responses = std::vector<json>{ result.data };
                }
            } else {
                // error received, ignore everything else
                res_error(res, result.data);
                return;
            }
        }

        // write JSON response
        json root = is_openai
            ? format_embeddings_response_oaicompat(body, responses, false)
            : responses[0];
        return res.set_content(root.dump(), "application/json; charset=utf-8");
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request & req, httplib::Response & res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.lora_adapters.size(); ++i) {
            auto & la = ctx_server.lora_adapters[i];
            result.push_back({
                {"id", i},
                {"path", la.path},
                {"scale", la.scale},
            });
        }
        res.set_content(result.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto handle_lora_adapters_apply = [&](const httplib::Request & req, httplib::Response & res) {
        const std::vector<json> body = json::parse(req.body);
        int max_idx = ctx_server.lora_adapters.size();

        // clear existing value
        for (auto & la : ctx_server.lora_adapters) {
            la.scale = 0.0f;
        }

        // set value
        for (auto entry : body) {
            int id      = entry.at("id");
            float scale = entry.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.lora_adapters[id].scale = scale;
            } else {
                throw std::runtime_error("invalid adapter id");
            }
        }

        server_task task;
        task.type = SERVER_TASK_TYPE_SET_LORA;
        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto list_saved_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        json response = json::array();
        namespace fs = std::filesystem;

        try {
            for (const auto& entry : fs::directory_iterator(params.slot_save_path)) {
                if (!entry.is_regular_file() || entry.file_size() < 12) {
                    continue;
                }

                std::ifstream file(entry.path(), std::ios::binary);
                if (!file) continue;

                uint32_t magic, version, n_token_count;
                file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                file.read(reinterpret_cast<char*>(&version), sizeof(version));
                file.read(reinterpret_cast<char*>(&n_token_count), sizeof(n_token_count));

                if (magic != LLAMA_STATE_SEQ_MAGIC ||
                    version != LLAMA_STATE_SEQ_VERSION ||
                    entry.file_size() < (12 + (n_token_count * sizeof(llama_token)))) {
                    continue;
                }

                std::vector<llama_token> tokens(n_token_count);
                file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(llama_token));

                //C++17 is not modern enough to have a nice and portable way to get the mtime of a file
                //so the following seems to be needed
                auto ftime = fs::last_write_time(entry.path());
                auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
                );
                std::time_t c_time = std::chrono::system_clock::to_time_t(system_time);
                std::tm tm_struct;
                #if defined(_WIN32)
                localtime_s(&tm_struct, &c_time);
                #else
                localtime_r(&c_time, &tm_struct);
                #endif
                std::ostringstream oss;
                oss << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
                auto str_time = oss.str();


                response.push_back({
                    {"filename", entry.path().filename().string()},
                    {"filesize", entry.file_size()},
                    {"mtime", str_time},
                    {"token_count", n_token_count},
                    {"prompt", tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend())}
                });
            }
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    const auto list_slot_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        json response = json::array();
        for (server_slot & slot : ctx_server.slots) {
            response.push_back({
                {"slot_id", slot.id},
                {"token_count", slot.cache_tokens.size()},
                {"prompt", slot.cache_tokens.detokenize(ctx_server.ctx, true) }
            });
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };


    const auto delete_saved_prompt = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res)-> void {
        json response;
        namespace fs = std::filesystem;

        try {
            const json body = json::parse(req.body);
            const std::string filename_str = body.at("filename");

            // prevent directory traversal attacks
            if (filename_str.find("..") != std::string::npos || filename_str.find('/') != std::string::npos || filename_str.find('\\') != std::string::npos) {
                res.status = 400;
                response = {{"error", "Invalid filename format."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            const fs::path file_to_delete = fs::path(params.slot_save_path) / fs::path(filename_str);

            if (!fs::exists(file_to_delete) || !fs::is_regular_file(file_to_delete)) {
                res.status = 404;
                response = {{"error", "File not found."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            if (fs::remove(file_to_delete)) {
                response = {
                    {"status", "deleted"},
                    {"filename", filename_str}
                };
            } else {
                res.status = 500;
                response = {{"error", "Failed to delete the file."}};
            }
        } catch (const json::parse_error& e) {
            res.status = 400;
            response = {{"error", "Invalid JSON request body."}};
        } catch (const json::out_of_range& e) {
            res.status = 400;
            response = {{"error", "Missing 'filename' key in request body."}};
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    const auto rename_saved_prompt = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res)-> void {
        json response;
        namespace fs = std::filesystem;

        try {
            const json body = json::parse(req.body);
            const std::string old_filename_str = body.at("old_filename");
            const std::string new_filename_str = body.at("new_filename");

            if (old_filename_str.find("..") != std::string::npos || old_filename_str.find_first_of("/\\") != std::string::npos ||
                new_filename_str.find("..") != std::string::npos || new_filename_str.find_first_of("/\\") != std::string::npos) {
                res.status = 400;
                response = {{"error", "Invalid filename format."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            const fs::path old_path = fs::path(params.slot_save_path) / old_filename_str;
            const fs::path new_path = fs::path(params.slot_save_path) / new_filename_str;

            if (!fs::exists(old_path) || !fs::is_regular_file(old_path)) {
                res.status = 404;
                response = {{"error", "Source file not found."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            if (fs::exists(new_path)) {
                res.status = 409;
                response = {{"error", "Destination filename already exists."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            std::error_code ec;
            fs::rename(old_path, new_path, ec);

            if (ec) {
                res.status = 500;
                response = {{"error", "Failed to rename file: " + ec.message()}};
            } else {
                response = {
                    {"status", "renamed"},
                    {"old_filename", old_filename_str},
                    {"new_filename", new_filename_str}
                };
            }

        } catch (const json::parse_error& e) {
            res.status = 400;
            response = {{"error", "Invalid JSON request body."}};
        } catch (const json::out_of_range& e) {
            res.status = 400;
            response = {{"error", "Missing 'old_filename' or 'new_filename' in request body."}};
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }

        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    auto handle_static_file = [](unsigned char * content, size_t len, const char * mime_type) {
        return [content, len, mime_type](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(content), len, mime_type);
            return false;
        };
    };
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    const auto handle_version = [&params, sqlite_extension_loaded](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            json{{"version", 4},
            {"features", {{"sql", !params.sql_save_file.empty()}, {"zstd_compression", sqlite_extension_loaded}}}}.dump(),
            "application/json"
        );
    };
#else
    const auto handle_version = [](const httplib::Request&, httplib::Response& res)-> void {
        res.set_content(
             json{{"version", 4},
             {"features", {{"sql", false}, {"zstd_compression", false}}}}.dump(),
             "application/json"
        );
    };
#endif

#ifdef SQLITE3_MODERN_CPP_SUPPORT
    auto db_handler = [db_handle](auto func) {
        return [func, db_handle](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
	    try {
                const json body = !req.body.empty() ? json::parse(req.body) : json::object();
                func(*db_handle, body, req, res);
            } catch(const std::exception& e) {
                res.status = 500;
                res.set_content(
                    json{{"ok", false}, {"message", e.what()}}.dump(),
                    "application/json"
                );
            }
        };
    };
#else
    auto db_handler = [db_handle](auto func) {
        return [func, db_handle](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.status = 500;
            res.set_content(
                json{{"ok", false}, {"message", "Sqlite3 support was not enabled. Recompile with '-DLLAMA_SERVER_SQLITE3=ON'"}}.dump(),
                "application/json"
            );
        };
    };
#endif

    const auto normalize_store_name = [](const std::string& storeName) {
        if(storeName.empty()) return std::string("sessions");

        std::string normalized;
        normalized.reserve(storeName.size());

        for(char c : storeName) {
            if(std::isalpha(static_cast<unsigned char>(c))) {
                normalized.push_back(std::tolower(static_cast<unsigned char>(c)));
            }
        }

        return normalized.empty() ? "sessions" : normalized;
    };

    const auto get_key_string = [](const json& j) {
        return j.is_string() ? j.get<std::string>() : j.dump();
    };


    const auto handle_load = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        std::string data;
	const std::string store = normalize_store_name(body["storeName"]);
	db.db << "SELECT data FROM " + store + " WHERE key = ?" << get_key_string(body["key"]) >> data;
	if(data.empty()) {
            res.status = 404;
            res.set_content(json{{"ok", false}, {"message", "Key not found"}}.dump(), "application/json");
        } else {
            json response{{"ok", true}};
	    response["result"] = (store == "names") ? json(data) : json::parse(data);
            res.set_content(response.dump(), "application/json");
        }
    });

    const auto handle_save = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        const std::string store = normalize_store_name(body["storeName"]);
        const std::string data = (store == "names") ? body["data"].get<std::string>() : body["data"].dump();
        db.db << "INSERT OR REPLACE INTO " + store + " (key, data) VALUES (?, ?)" << get_key_string(body["key"]) << data;
        res.set_content(json{{"ok", true}, {"result", "Data saved successfully"}}.dump(), "application/json");
    });

    const auto handle_rename = db_handler([get_key_string](auto& db, const json& body, auto&, auto& res) {
        db.db << "UPDATE names SET data = ? WHERE key = ?"
            << body["newName"].get<std::string>()
            << get_key_string(body["key"]);
        res.set_content(json{{"ok", true}, {"result", "Session renamed successfully"}}.dump(), "application/json");
    });

    const auto handle_all = db_handler([normalize_store_name](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT key, data FROM " + normalize_store_name(body["storeName"]) >>
            [&](const std::string& key, const std::string& data) {
                result[key] = json::parse(data);
            };
        res.set_content(json{{"ok", true}, {"result", result}}.dump(), "application/json");
    });

    const auto handle_sessions = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT key, data FROM names" >> [&](const std::string& key, const std::string& data) {
            result[key] = data;
        };
        res.set_content(json{{"ok", true}, {"result", result}}.dump(), "application/json");
    });

    const auto handle_delete = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        db.db << "DELETE FROM " + normalize_store_name(body["storeName"]) + " WHERE key = ?"
            << get_key_string(body["key"]);
        res.set_content(json{{"ok", true}, {"result", "Session deleted successfully"}}.dump(), "application/json");
    });

    const auto handle_vacuum = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "VACUUM";
        res.set_content(json{"ok", true}.dump(), "application/json");
    });

    const auto handle_zstd_get_configs = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT id, config FROM _zstd_configs" >> [&](const std::string id, const std::string& config) {
            result[id] = config;
        };
        res.set_content(json{{"ok", true}, {"configs", result}}.dump(), "application/json");
    });

    const auto handle_zstd_maintenance = db_handler([](auto& db, const json& body, auto&, auto& res) {
        std::string data;
        if (body["duration"].is_null()) {
            db.db << "select zstd_incremental_maintenance(?, ?)" <<  nullptr << body["db_load"].get<double>() >> data;
        }
	else {
            db.db << "select zstd_incremental_maintenance(?, ?)" << body["duration"].get<double>() << body["db_load"].get<double>() >> data;
        }
        json response{{"ok", true}};
        response["result"] = json::parse(data);
        res.set_content(response.dump(), "application/json");
    });

    const auto handle_zstd_enable = db_handler([](auto& db, const json& body, auto&, auto& res) {
        db.db << "select zstd_enable_transparent('{\"table\": \"" + body["table"].get<std::string>() + "\",\"column\": \"" + body["column"].get<std::string>() + "\", \"compression_level\": " + std::to_string(body["compression_level"].get<int>()) + ", \"dict_chooser\": \"''a''\", \"train_dict_samples_ratio\": " + std::to_string(body["train_dict_samples_ratio"].get<int>()) + "}')";
        res.set_content(json{"ok", true}.dump(), "application/json");
    });

    const auto handle_zstd_config_update = db_handler([](auto& db, const json& body, auto&, auto& res) {
        std::string patch_json = "{\"compression_level\": " + std::to_string(body["compression_level"].get<int>()) + ", \"train_dict_samples_ratio\": " + std::to_string(body["train_dict_samples_ratio"].get<int>()) + "}";
        db.db << "update _zstd_configs set config = json_patch(config, '" + patch_json + "')";
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    //
    // Router
    //
    if (params.webui == COMMON_WEBUI_NONE) {
        LLAMA_LOG_INFO("Web UI is disabled\n");
    }
    else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            svr->set_base_dir(params.public_path);
        }

        {
            // register static assets routes
            if (!params.public_path.empty()) {
                // Set the base directory for serving static files
                bool is_found = svr->set_mount_point("/", params.public_path);
                if (!is_found) {
                    GGML_ABORT("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                    return 1;
                }
            }
            else {

                // using embedded static index.html
                svr->Get("/", [params](const httplib::Request& req, httplib::Response& res) {
                    if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                        res.set_content("Error: gzip is not supported by this browser", "text/plain");
                    }
                    else {
                        res.set_header("Content-Encoding", "gzip");
                        // COEP and COOP headers, required by pyodide (python interpreter)
                        res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                        res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                        if (params.webui == COMMON_WEBUI_AUTO) {
                            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                        }
                        else if (params.webui == COMMON_WEBUI_LLAMACPP) {
                            res.set_content(reinterpret_cast<const char*>(index_llamacpp_html_gz), index_llamacpp_html_gz_len, "text/html; charset=utf-8");
                        }
                        else {
                            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                        }
                    }
                    return false;
                    });
            }
        }
    }
    // register API routes
    svr->Get ("/health",              handle_health);
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Get("/v1/props",             handle_props_simple);
    svr->Get ("/v1/models",           handle_models);
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions", handle_completions); // legacy
    svr->Post("/v1/completions",     handle_completions_oai);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    svr->Post("/apply-template",      handle_apply_template);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    svr->Get ("/slots/list",          list_slot_prompts);
    if (!params.slot_save_path.empty()) {
        // these endpoints rely on slot_save_path existing
        svr->Post("/slots/:id_slot",  handle_slots_action);
        svr->Get ("/list",            list_saved_prompts);
        svr->Post("/delete_prompt",   delete_saved_prompt);
        svr->Post("/rename_prompt",   rename_saved_prompt);

    }

    svr->Get ("/version", handle_version);
    if (!params.sql_save_file.empty()) {
        // these endpoints rely on sql_save_file existing
        svr->Post("/load", handle_load);
        svr->Post("/save", handle_save);
        svr->Post("/rename", handle_rename);
        svr->Post("/all", handle_all);
        svr->Post("/sessions", handle_sessions);
        svr->Get ("/sessions", handle_sessions);
        svr->Post("/delete", handle_delete);
        //VACUUM is there for the extension but does not require the extension
        svr->Get ("/vacuum", handle_vacuum);
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        if (sqlite_extension_loaded) {
            svr->Get ("/zstd_get_configs", handle_zstd_get_configs);
            svr->Post("/zstd_incremental_maintenance", handle_zstd_maintenance);
            svr->Post("/zstd_enable_transparent", handle_zstd_enable);
            svr->Post("/zstd_update_transparent", handle_zstd_config_update);
	}
#endif
    }
    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    LOG_INFO("HTTP server listening", log_data);

    // run the HTTP server in a thread - see comment below
    std::thread t([&]() {
        if (!svr->listen_after_bind()) {
            state.store(SERVER_STATE_ERROR);
            return 1;
        }

        return 0;
    });

    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
        });
    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots, &ctx_server));
    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));

    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    ctx_server.queue_tasks.start_loop();

    svr->stop();
    t.join();

    llama_backend_free();

    return 0;
}
