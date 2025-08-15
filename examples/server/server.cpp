#pragma warning(disable : 4996)
#include "utils.hpp"

#include "common.h"
#include "speculative.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "grammar-parser.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// disable Nagle's algorithm
#define CPPHTTPLIB_TCP_NODELAY true
#include "httplib.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
#include "index.html.gz.hpp"
#include "loading.html.hpp"
#include "function_calls.hpp"
#include "streaming_chat.hpp"
#include "../../common/chat-parser.h"

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

using json = nlohmann::ordered_json;

bool server_verbose = false;
bool server_log_json = true;



enum stop_type {
    STOP_TYPE_FULL,
    STOP_TYPE_PARTIAL,
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
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
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

    server_task_type type;
    json data;

    bool infill    = false;
    bool embedding = false;
};

struct server_task_result {
    int id       = -1;
    int id_multi = -1;

    json data;

    bool stop;
    bool error;
    result_timings timings;

};

std::unordered_map<int, server_task_result > server_task_result_dict = {};

// Helper functions for content cleaning
static std::string remove_simple_function_calls(const std::string& content) {
    std::string cleaned = content;
    const std::string func_pattern = "functions.";
    size_t pos = 0;
    while ((pos = cleaned.find(func_pattern, pos)) != std::string::npos) {
        size_t func_start = pos;

        // Find the opening brace for arguments
        size_t brace_pos = cleaned.find('{', pos);
        if (brace_pos == std::string::npos) {
            pos += func_pattern.length();
            continue;
        }

        // Find the matching closing brace
        int brace_count = 1;
        size_t end_pos = brace_pos + 1;
        while (end_pos < cleaned.length() && brace_count > 0) {
            if (cleaned[end_pos] == '{') brace_count++;
            else if (cleaned[end_pos] == '}') brace_count--;
            end_pos++;
        }

        if (brace_count == 0) {
            // Remove the entire function call
            cleaned.erase(func_start, end_pos - func_start);
            pos = func_start;
        } else {
            pos += func_pattern.length();
        }
    }
    return cleaned;
}

static std::string remove_xml_function_calls(const std::string& content) {
    std::string cleaned = content;
    size_t pos = 0;
    while ((pos = cleaned.find("<tool_call>", pos)) != std::string::npos) {
        size_t tool_call_start = pos;
        size_t tool_call_end = cleaned.find("</tool_call>", tool_call_start);
        if (tool_call_end == std::string::npos) {
            pos = tool_call_start + 11;
            continue;
        }

        // Remove the entire XML tool call block
        cleaned.erase(tool_call_start, tool_call_end - tool_call_start + 12);
        pos = tool_call_start;
    }
    return cleaned;
}

static std::string clean_all_function_call_formats(const std::string& content) {
    std::string cleaned = content;

    // Remove XML format first
    cleaned = remove_xml_function_calls(cleaned);

    // Then remove simple format
    cleaned = remove_simple_function_calls(cleaned);

    // Trim whitespace from cleaned content
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);

    return cleaned;
}

struct server_task_multi {
    int id = -1;

    std::set<int> subtasks_remaining;
    std::vector<server_task_result> results;
};

struct slot_params {
    bool stream       = true;
    bool cache_prompt = true; // remember the prompt to avoid reprocessing all prompt

    int32_t  n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t  n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t  n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    bool timings_per_token = false;
    json input_prefix;
    json input_suffix;

    // speculative decoding parameters
    struct {
        int n_max = 16;  // max drafted tokens
        int n_min = 0;  // min drafted tokens to accept
        float p_min = 0.75f; // min probability required to accept a token in the draft
    } speculative;
};

struct server_slot {
    int id;
    int id_task = -1;
    int id_multi = -1;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;
    slot_command command = SLOT_COMMAND_NONE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt; // can be either a string, array of strings or array of token ids

    // when a task is submitted, we first tokenize the prompt and store it here
    std::vector<llama_token> prompt_tokens;

    std::string generated_text;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    // Streaming tool call state
    ik_chat_msg previous_msg;
    ik_chat_msg current_msg;
    std::vector<std::string> tool_call_ids;

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

    // sampling
    llama_token sampled;
    struct llama_sampling_params sparams;
    llama_sampling_context * ctx_sampling = nullptr;
    json json_schema;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

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

        generated_token_probs.clear();

        // Reset streaming tool call state
        previous_msg = ik_chat_msg();
        current_msg = ik_chat_msg();
        tool_call_ids.clear();

        // Reset speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
    }

    // Update chat message and compute diffs for streaming tool calls
    // Based on original llama.cpp update_chat_msg pattern
    const ik_chat_msg & update_chat_msg(std::vector<ik_chat_msg_diff> & diffs) {
        ik_chat_msg previous = current_msg;

        try {
            // Parse generated text incrementally (is_partial = true during generation)
            bool is_partial = !stopped_eos && !stopped_word && !stopped_limit;
            ik_chat_msg new_msg = parse_chat_message_incremental(generated_text, is_partial, oaicompat_model);

            if (!new_msg.empty()) {
                // Ensure tool call IDs are set consistently across streaming chunks
                new_msg.ensure_tool_call_ids_set(tool_call_ids, generate_tool_call_id);
                current_msg = new_msg;

                // Compute diffs for streaming
                diffs = ik_chat_msg_diff::compute_diffs(previous, current_msg);
            }
        } catch (const std::exception& e) {
            // If parsing fails, don't update current_msg and return empty diffs
            diffs.clear();
        }

        return current_msg;
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
        };
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.prompt_n = n_prompt_tokens_processed;
        timings.prompt_ms = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;


        timings.predicted_n = n_decoded;
        timings.predicted_ms = (ggml_time_us() - t_start_generation) / 1e3;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }
    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, const stop_type type) {
        size_t stop_pos = std::string::npos;

        for (const std::string & word : params.antiprompt) {
            size_t pos;

            if (type == STOP_TYPE_FULL) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                pos = find_partial_stop_string(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (type == STOP_TYPE_FULL) {
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
    std::function<void(server_task       &)> callback_new_task;
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
    void defer(server_task task) {
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
    void on_new_task(std::function<void(server_task &)> callback) {
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
                server_task task = queue_tasks.front();
                queue_tasks.erase(queue_tasks.begin());
                lock.unlock();
                LOG_VERBOSE("callback_new_task", {{"id_task", task.id}});
                callback_new_task(task);
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

    server_metrics metrics;

    common_chat_templates chat_templates;
    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    ~server_context() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }

        if (model) {
            llama_free_model(model);
            model = nullptr;
        }

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

        // dedicate one sequence to the system prompt
        params.n_parallel += 1;

        llama_init_result llama_init = llama_init_from_gpt_params(params);

        model = llama_init.model;
        ctx = llama_init.context;
        lora_adapters = llama_init.lora_adapters;
        params.n_parallel -= 1; // but be sneaky about it
        if (model == nullptr) {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_should_add_bos_token(model);
        GGML_ASSERT(llama_add_eos_token(model) != 1);

        if (params.chat_template.empty() && !validate_model_chat_template(params.use_jinja)) {
            LOG_WARNING("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
            chat_templates = llama_chat_templates_from_model(model, "chatml");
        }
        else {
            chat_templates = llama_chat_templates_from_model(model, params.chat_template);
        }
        GGML_ASSERT(chat_templates.template_default.get() != nullptr);
      
        // Load draft model for speculative decoding if specified
        if (!params.model_draft.empty()) {
            LOG_INFO("loading draft model", {{"model", params.model_draft}});

            gpt_params params_dft;
            params_dft.model = params.model_draft;
            params_dft.n_ctx = params.n_ctx_draft == 0 ? params.n_ctx / params.n_parallel : params.n_ctx_draft;
            params_dft.n_gpu_layers = params.n_gpu_layers_draft;
            params_dft.n_parallel = 1;
            params_dft.cache_type_k = params.cache_type_k_draft.empty() ? params.cache_type_k : params.cache_type_k_draft;
            params_dft.cache_type_v = params.cache_type_v_draft.empty() ? params.cache_type_v : params.cache_type_v_draft;
            params_dft.flash_attn = params.flash_attn;

            llama_init_result llama_init_dft = llama_init_from_gpt_params(params_dft);

            llama_model * model_dft = llama_init_dft.model;
            if (model_dft == nullptr) {
                LOG_ERROR("failed to load draft model", {{"model", params.model_draft}});
                return false;
            }

            if (!llama_speculative_are_compatible(ctx, llama_init_dft.context)) {
                LOG_ERROR("the draft model is not compatible with the target model", {});
                return false;
            }

            const int n_ctx_dft = llama_n_ctx(llama_init_dft.context);

            cparams_dft = llama_context_params_from_gpt_params(params_dft);
            cparams_dft.n_batch = n_ctx_dft;

            model_draft = llama_init_dft.model;
            ctx_draft = llama_init_dft.context;
        }
        return true;
    }

    bool validate_model_chat_template(bool use_jinja) const {
        llama_chat_message chat[] = {{"user", "test"}};


        if (use_jinja) {
            auto templates = llama_chat_templates_from_model(model, "");
            GGML_ASSERT(templates.template_default);
            try {
                templates.template_default->apply({ {
                    {"role", "user"},
                    {"content", "test"},
                } }, json(), true);
                if (templates.template_tool_use) {
                    templates.template_tool_use->apply({ {
                        {"role", "user"},
                        {"content", "test"},
                    } }, json(), true);
    }
                return true;
            }
            catch (const std::exception& e) {
                LOG_ERROR("failed to apply template: %s\n", e.what());
                return false;
            }
        }
        else {
            const char* tmpl = llama_model_chat_template(model, /* name */ nullptr);
            const int32_t chat_res = llama_chat_apply_template(model, tmpl, chat, 1, true, nullptr, 0);
            return chat_res > 0;
        }
    }

    void init() {
        const int32_t n_ctx_slot = n_ctx / params.n_parallel;

        LOG_INFO("initializing slots", {{"n_slots", params.n_parallel}});

        for (int i = 0; i < params.n_parallel; i++) {
            server_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;

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

                slot.ctx_dft = llama_new_context_with_model(model_draft, cparams_dft);
                if (slot.ctx_dft == nullptr) {
                    LOG_ERROR("failed to create draft context", {});
                    return;
                }

                slot.spec = llama_speculative_init(slot.ctx_dft);
                if (slot.spec == nullptr) {
                    LOG_ERROR("failed to create speculator", {});
                    return;
                }
            }

            slot.reset();

            slots.push_back(slot);
        }

        default_generation_settings_for_props = get_formated_generation(slots.front());
        default_generation_settings_for_props["seed"] = -1;

        // the update_slots() logic will always submit a maximum of n_batch tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);

            // only a single seq_id per token is needed
            batch = llama_batch_init(n_batch, 0, 1);
        }

        metrics.init();
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

    server_slot * get_available_slot(const std::string & prompt) {
        server_slot * ret = nullptr;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f && !prompt.empty()) {
            int max_lcp_len = 0;
            float similarity = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (!slot.available()) {
                    continue;
                }

                // skip the slot if it does not contains prompt
                if (!slot.prompt.is_string()) {
                    continue;
                }

                // current slot's prompt
                std::string slot_prompt = slot.prompt.get<std::string>();

                // length of the current slot's prompt
                int slot_prompt_len = slot_prompt.size();

                // length of the Longest Common Prefix between the current slot's prompt and the input prompt
                int lcp_len = common_part(slot_prompt, prompt);

                // fraction of the common substring length compared to the current slot's prompt length
                similarity = static_cast<float>(lcp_len) / slot_prompt_len;

                // select the current slot if the criteria match
                if (lcp_len > max_lcp_len && similarity > slot_prompt_similarity) {
                    max_lcp_len = lcp_len;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                LOG_VERBOSE("selected slot by lcp similarity", {
                    {"id_slot", ret->id},
                    {"max_lcp_len", max_lcp_len},
                    {"similarity", similarity},
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

        return ret;
    }

    bool launch_slot_with_task(server_slot & slot, const server_task & task) {
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
        if (data.contains("json_schema") && !data.at("json_schema").is_null() && data.contains("grammar") && !data.at("grammar").is_null()) {
            send_error(task, "Either \"json_schema\" or \"grammar\" can be specified, but not both", ERROR_TYPE_INVALID_REQUEST);
            return false;
        } else if (data.contains("json_schema") && !data.contains("grammar")) {
            try {
                auto schema                = json_value(data, "json_schema", json::object());
                slot.sparams.grammar       = json_schema_to_grammar(schema);
            } catch (const std::exception & e) {
                send_error(task, std::string("\"json_schema\": ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        } else {
            slot.sparams.grammar       = json_value(data, "grammar",           default_sparams.grammar);
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
            const auto & prompt = data.find("prompt");
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
            slot.sparams.logit_bias.clear();

            if (json_value(data, "ignore_eos", false)) {
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
        slot.prompt_tokens.clear();

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

            llama_batch_clear(batch);

            for (int i = 0; i < (int)system_tokens.size(); ++i) {
                llama_batch_add(batch, system_tokens[i], i, { 0 }, false);
            }

            const int32_t n_batch = llama_n_batch(ctx);

            for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
                const int32_t n_tokens = std::min(params.n_batch, batch.n_tokens - i);
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

                if (llama_decode(ctx, batch_view) != 0) {
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
        const std::string token_str = llama_token_to_piece(ctx, result.tok, params.special);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        if (slot.ctx_sampling->params.use_penalty_prompt_tokens && result.tok != -1) {
            // we can change penalty_prompt_tokens because it is always created from scratch each request
            slot.ctx_sampling->params.penalty_prompt_tokens.push_back(result.tok);
        }

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i) {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80) {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0) {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_FULL);
            if (stop_pos != std::string::npos) {
                is_stop_full = true;
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else {
                is_stop_full = false;
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), STOP_TYPE_PARTIAL);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!slot.has_next_token && !is_stop_full && stop_pos > 0)) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
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

    json get_formated_generation(const server_slot & slot) const {
        const auto eos_bias   =             slot.sparams.logit_bias.find(llama_token_eos(model));
        const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() && eos_bias->second < 0.0f && std::isinf(eos_bias->second);

        std::vector<std::string> samplers_sequence;
        samplers_sequence.reserve(slot.sparams.samplers_sequence.size());
        for (const auto & sampler_type : slot.sparams.samplers_sequence) {
            samplers_sequence.emplace_back(llama_sampling_type_to_str(sampler_type));
        }

        return json {
            {"n_ctx",                     slot.n_ctx},
            {"n_predict",                 slot.n_predict},
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
            {"n_predict",                 slot.params.n_predict}, // TODO: fix duplicate key n_predict
            {"n_keep",                    slot.params.n_keep},
            {"n_discard",                 slot.params.n_discard},
            {"ignore_eos",                ignore_eos},
            {"stream",                    slot.params.stream},
            {"logit_bias",                slot.sparams.logit_bias},
            {"n_probs",                   slot.sparams.n_probs},
            {"min_keep",                  slot.sparams.min_keep},
            {"grammar",                   slot.sparams.grammar},
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

    void send_partial_response(server_slot & slot, completion_token_output tkn) {
        server_task_result res;
        res.id       = slot.id_task;
        res.id_multi = slot.id_multi;
        res.error    = false;
        res.stop     = false;

        // Update chat message and compute diffs for streaming tool calls
        // Following original llama.cpp pattern (server.cpp:2503)
        std::vector<ik_chat_msg_diff> oaicompat_msg_diffs;
        slot.update_chat_msg(oaicompat_msg_diffs);

        // For text completion endpoints, send actual content; for chat completion, use diffs
        // OpenAI-compatible chat endpoints use empty content with diffs for tool calls
        res.data     = json {
            {"content",    slot.oaicompat ? "" : tkn.text_to_send},  // Text completion needs actual content
            {"stop",       false},
            {"id_slot",    slot.id},
            {"multimodal", false}
        };

        // Store diffs for format_partial_response_oaicompat to use
        // Convert ik_chat_msg_diff to JSON format for storage
        json diffs_json = json::array();
        for (const auto & diff : oaicompat_msg_diffs) {
            json diff_obj;
            if (!diff.content_delta.empty()) {
                diff_obj["content_delta"] = diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                diff_obj["tool_call_index"] = diff.tool_call_index;
                diff_obj["tool_call_delta"] = {
                    {"id", diff.tool_call_delta.id},
                    {"name", diff.tool_call_delta.name},
                    {"arguments", diff.tool_call_delta.arguments}
                };
            }
            if (!diff_obj.empty()) {
                diffs_json.push_back(diff_obj);
            }
        }
        res.data["oaicompat_msg_diffs"] = diffs_json;

        if (slot.sparams.n_probs > 0) {
            const std::vector<llama_token> to_send_toks = llama_tokenize(ctx, tkn.text_to_send, false);
            const size_t probs_pos      = std::min(slot.n_sent_token_probs,                       slot.generated_token_probs.size());
            const size_t probs_stop_pos = std::min(slot.n_sent_token_probs + to_send_toks.size(), slot.generated_token_probs.size());

            std::vector<completion_token_output> probs_output;
            if (probs_pos < probs_stop_pos) {
                probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin() + probs_pos,
                        slot.generated_token_probs.begin() + probs_stop_pos);
            }
            slot.n_sent_token_probs = probs_stop_pos;

            res.data["completion_probabilities"] = probs_vector_to_json(ctx, probs_output);
        }

        if (slot.oaicompat) {
            res.data["oaicompat_token_ctr"] = slot.n_decoded;
            res.data["model"] = slot.oaicompat_model;
        }
        // populate timings if this is final response or timings_per_token is enabled
        if (slot.params.timings_per_token) {
            //res.data["timings"] = slot.get_formated_timings();
            res.timings = slot.get_timings();
        }
        server_task_result_dict[slot.id_task] = res;
        queue_results.send(std::move(res));
    }

    void send_final_response(const server_slot & slot) {
        server_task_result res;
        res.id       = slot.id_task;
        res.id_multi = slot.id_multi;
        res.error    = false;
        res.stop     = true;
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
            {"timings",             slot.get_formated_timings()}
        };

        if (slot.sparams.n_probs > 0) {
            std::vector<completion_token_output> probs;
            if (!slot.params.stream && slot.stopped_word) {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                probs = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                probs = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }

            res.data["completion_probabilities"] = probs_vector_to_json(ctx, probs);
        }

        if (slot.oaicompat) {
            res.data["oaicompat_token_ctr"] = slot.n_decoded;
            res.data["model"] = slot.oaicompat_model;
        }

        queue_results.send(res);
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
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id + 1) {
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
                };

                continue;
            }

            llama_embd_normalize(embd, embd_res.data(), n_embd);

            res.data = json {
                {"embedding", embd_res},
            };
        }

        queue_results.send(res);
    }

    void request_completion(int id_task, int id_multi, json data, bool infill, bool embedding) {
        server_task task;
        task.id        = id_task;
        task.id_multi  = id_multi;
        task.id_target = 0;
        task.data      = std::move(data);
        task.infill    = infill;
        task.embedding = embedding;
        task.type      = SERVER_TASK_TYPE_COMPLETION;

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
                queue_tasks.post(task);
            } else {
                split_multiprompt_task(id_task, task);
            }
        } else {
            queue_tasks.post(task);
        }
    }

    void request_cancel(int id_task) {
        server_task task;
        task.type      = SERVER_TASK_TYPE_CANCEL;
        task.id_target = id_task;

        queue_tasks.post(task);
    }

    void split_multiprompt_task(int id_multi, const server_task & multiprompt_task) {
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
            request_completion(subtask_ids[i], id_multi, subtask_data, multiprompt_task.infill, multiprompt_task.embedding);
        }
    }

    void process_single_task(const server_task & task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
                {
                    const int id_slot = json_value(task.data, "id_slot", -1);

                    server_slot * slot;

                    if (id_slot != -1) {
                        slot = get_slot_by_id(id_slot);
                    } else {
                        std::string prompt;
                        if (task.data.contains("prompt") && task.data.at("prompt").is_string()) {
                            prompt = json_value(task.data, "prompt", std::string());
                        }

                        slot = get_available_slot(prompt);
                    }

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        LOG_VERBOSE("no slot is available", {{"id_task", task.id}});
                        queue_tasks.defer(task);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(task);
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
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(task);
                        break;
                    }

                    const size_t token_count = slot->cache_tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.data.at("filename");
                    std::string filepath = task.data.at("filepath");

                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), token_count);

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
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(task);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.data.at("filename");
                    std::string filepath = task.data.at("filepath");

                    slot->cache_tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
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
                    int id_slot = task.data.at("id_slot");
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!slot->available()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        LOG_VERBOSE("requested slot is unavailable", {{"id_task", task.id}});
                        queue_tasks.defer(task);
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

            queue_tasks.post(task);
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.ga_n == 1) {
                if (slot.is_processing() && (int) system_tokens.size() + slot.n_past >= slot.n_ctx - 1) {
                    // Shift context
                    const int n_keep    = slot.params.n_keep + add_bos_token;
                    const int n_left    = (int) system_tokens.size() + slot.n_past - n_keep;
                    const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

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

                    llama_kv_cache_seq_rm (ctx, slot.id + 1, n_keep            , n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, slot.id + 1, n_keep + n_discard, system_tokens.size() + slot.n_past, -n_discard);

                    if (slot.params.cache_prompt) {
                        for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++) {
                            slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                        }

                        slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
                    }

                    slot.n_past -= n_discard;

                    slot.truncated = true;
                }
            }
        }

        // start populating the batch for this iteration
        llama_batch_clear(batch);

        // frist, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_IDLE) {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            const int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

            // TODO: we always have to take into account the "system_tokens"
            //       this is not great and needs to be improved somehow
            llama_batch_add(batch, slot.sampled, system_tokens.size() + slot_npast, { slot.id + 1 }, true);

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
                    if (prompt_tokens.empty()) {
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

                            prompt_tokens = embd_inp;
                        } else {
                            prompt_tokens = tokenize(slot.prompt, system_prompt.empty()); // add BOS if there isn't system prompt
                        }

                        slot.n_past = 0;
                        slot.n_prompt_tokens = prompt_tokens.size();

                        LOG_VERBOSE("prompt tokenized", {
                            {"id_slot",         slot.id},
                            {"id_task",         slot.id_task},
                            {"n_ctx",           slot.n_ctx},
                            {"n_keep",          slot.params.n_keep},
                            {"n_prompt_tokens", slot.n_prompt_tokens},
                            {"prompt_tokens",   tokens_to_str(ctx, prompt_tokens.cbegin(), prompt_tokens.cend())},
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
                            if (slot.params.n_keep < 0) {
                                slot.params.n_keep = slot.n_prompt_tokens;
                            }
                            slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                            // if input prompt is too big, truncate it (if group attention self-extend is disabled)
                            if (slot.ga_n == 1 && slot.n_prompt_tokens >= slot.n_ctx) {
                                const int n_left = slot.n_ctx - slot.params.n_keep;

                                const int n_block_size = n_left / 2;
                                const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                                std::vector<llama_token> new_tokens(
                                        prompt_tokens.begin(),
                                        prompt_tokens.begin() + slot.params.n_keep);

                                new_tokens.insert(
                                        new_tokens.end(),
                                        prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                        prompt_tokens.end());

                                prompt_tokens = std::move(new_tokens);

                                slot.truncated = true;
                                slot.n_prompt_tokens = prompt_tokens.size();

                                LOG_VERBOSE("input truncated", {
                                    {"id_slot",         slot.id},
                                    {"id_task",         slot.id_task},
                                    {"n_ctx",           slot.n_ctx},
                                    {"n_keep",          slot.params.n_keep},
                                    {"n_left",          n_left},
                                    {"n_prompt_tokens", slot.n_prompt_tokens},
                                    {"prompt_tokens",   tokens_to_str(ctx, prompt_tokens.cbegin(), prompt_tokens.cend())},
                                });

                                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                            }

                            llama_sampling_reset(slot.ctx_sampling);

                            if (!slot.params.cache_prompt) {
                                slot.n_past_se = 0;
                                slot.ga_i      = 0;
                            } else {
                                GGML_ASSERT(slot.ga_n == 1);

                                // reuse any previously computed tokens that are common with the new prompt
                                slot.n_past = common_part(slot.cache_tokens, prompt_tokens);

                                // push the prompt into the sampling context (do not apply grammar)
                                for (int i = 0; i < slot.n_past; ++i) {
                                    llama_sampling_accept(slot.ctx_sampling, ctx, slot.cache_tokens[i], false);
                                }
                            }
                        }

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            LOG_INFO("we have to evaluate at least 1 token to generate logits", {
                                { "id_slot", slot.id },
                                { "id_task", slot.id_task }
                            });

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
                    int p0 = (int) system_tokens.size() + slot.n_past;
                    if (!llama_kv_cache_seq_rm(ctx, slot.id + 1, p0, -1)) {
                        // could not partially delete (likely using a non-Transformer model)
                        llama_kv_cache_seq_rm(ctx, slot.id + 1, -1, -1);

                        p0 = (int) system_tokens.size();
                        if (p0 != 0) {
                            // copy over the system prompt when there is one
                            llama_kv_cache_seq_cp(ctx, 0, slot.id + 1, -1, -1);
                        }

                        // there is no common part left (except for the system prompt)
                        slot.n_past = 0;
                        slot.n_past_se = 0;
                        slot.ga_i = 0;
                        // TODO: is the system prompt ever in the sampling context?
                        llama_sampling_reset(slot.ctx_sampling);
                    }

                    // remove the non-common part from the cache
                    slot.cache_tokens.resize(slot.n_past);

                    LOG_INFO("kv cache rm [p0, end)", {
                        { "id_slot", slot.id },
                        { "id_task", slot.id_task },
                        { "p0",      p0 }
                    });

                    int32_t slot_npast = slot.n_past_se > 0 ? slot.n_past_se : slot.n_past;

                    int32_t ga_i = slot.ga_i;
                    int32_t ga_n = slot.ga_n;
                    int32_t ga_w = slot.ga_w;

                    // add prompt tokens for processing in the current batch
                    // TODO: the self-extend stuff here is a mess - simplify and/or abstract it somehow
                    for (; slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch; ++slot.n_past) {
                        if (slot.ga_n != 1) {
                            while (slot_npast >= ga_i + ga_w) {
                                const int bd = (ga_w/ga_n)*(ga_n - 1);
                                slot_npast -= bd;
                                ga_i += ga_w/ga_n;
                            }
                        }

                        llama_batch_add(batch, prompt_tokens[slot.n_past], system_tokens.size() + slot_npast, { slot.id + 1 }, false);

                        if (slot.params.cache_prompt) {
                            slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
                        }

                        slot.n_prompt_tokens_processed++;
                        slot_npast++;
                    }

                    LOG_VERBOSE("prompt processing progress", {
                        {"id_slot",  slot.id},
                        {"n_past",   slot.n_past},
                        {"n_ctx",    n_ctx},
                        {"n_tokens", batch.n_tokens},
                        {"progress", (float) slot.n_prompt_tokens_processed / slot.n_prompt_tokens},
                    });

                    // entire prompt has been processed - start decoding new tokens
                    if (slot.n_past == slot.n_prompt_tokens) {
                        slot.state   = SLOT_STATE_PROCESSING;
                        slot.command = SLOT_COMMAND_NONE;

                        GGML_ASSERT(batch.n_tokens > 0);
                        llama_sampling_reset(slot.ctx_sampling);
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

                        llama_kv_cache_seq_add(ctx, slot.id + 1, slot.ga_i, slot.n_past_se, ib * bd);
                        llama_kv_cache_seq_div(ctx, slot.id + 1, slot.ga_i + ib * bd, slot.ga_i + ib * bd + slot.ga_w, slot.ga_n);
                        llama_kv_cache_seq_add(ctx, slot.id + 1, slot.ga_i + ib * bd + slot.ga_w, slot.n_past_se + ib * bd, dd);

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
                const llama_token id = llama_sampling_sample(slot.ctx_sampling, ctx, NULL, slot.i_batch - i);

                llama_sampling_accept(slot.ctx_sampling, ctx, id, true);

                slot.n_decoded += 1;
                if (slot.n_decoded == 1) {
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                llama_token_data_array cur_p = { slot.ctx_sampling->cur.data(), slot.ctx_sampling->cur.size(), false };
                result.tok = id;

                const size_t n_probs = std::min(cur_p.size, (size_t) slot.sparams.n_probs);
                if (n_probs > 0) {
                    const size_t n_valid = slot.ctx_sampling->n_valid;

                    // Make sure at least n_probs top tokens are at the front of the vector:
                    if (slot.sparams.temp == 0.0f && n_probs > n_valid) {
                        llama_sample_top_k(ctx, &cur_p, n_probs, 0);
                    }

                    if (slot.sparams.temp == 0.0f) {
                        // With greedy sampling the probabilities have possibly not been calculated.
                        for (size_t i = 0; i < n_probs; ++i) {
                            result.probs.push_back({
                                cur_p.data[i].id,
                                i == 0 ? 1.0f : 0.0f
                            });
                        }
                    } else {
                        for (size_t i = 0; i < n_probs; ++i) {
                            result.probs.push_back({
                                cur_p.data[i].id,
                                i >= n_valid ? 0.0f : cur_p.data[i].p // Tokens filtered out due to e.g. top_k have 0 probability.
                            });
                        }
                    }
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

                const std::vector<llama_token> & cached_text_tokens = slot.cache_tokens;
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
                llama_batch_add(slot.batch_spec, id, slot.n_past, { slot.id + 1 }, true);

                for (size_t i = 0; i < draft.size(); ++i) {
                    llama_batch_add(slot.batch_spec, draft[i], slot.n_past + 1 + i, { slot.id + 1 }, true);
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
                slot.cache_tokens.insert(slot.cache_tokens.end(), ids.begin(), ids.end() - 1);

                llama_kv_cache_seq_rm(ctx, slot.id + 1, slot.n_past, -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok = ids[i];
                    result.text_to_send = llama_token_to_piece(ctx, result.tok, params.special);
                    // result.prob = 1.0f; // set later

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

    // Parse tool calls using model-specific format detection
    std::string model_name = json_value(request, "model", std::string(""));

    // Use the same parsing logic as streaming path for consistency
    ik_chat_msg parsed_msg = parse_chat_message_incremental(content, false, model_name);

    // Convert to JSON format for compatibility
    json tool_calls = json::array();
    for (const auto & tc : parsed_msg.tool_calls) {
        tool_calls.push_back({
            {"type", "function"},
            {"function", {
                {"name", tc.name},
                {"arguments", tc.arguments}
            }},
            {"id", tc.id}
        });
    }

    bool has_tool_calls = !tool_calls.empty();

    // Use cleaned content from parser (following original llama.cpp pattern)
    if (has_tool_calls) {
        content = parsed_msg.content; // Parser already cleaned the content
    }

    std::string finish_reason = "length";
    if (has_tool_calls) {
        finish_reason = "tool_calls";
    } else if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json message = json{{"role", "assistant"}};
    // Follow EXACT original llama.cpp pattern: content is null only when content is empty AND tool calls exist
    if (content.empty() && has_tool_calls) {
        message["content"] = nullptr;  // Original: json() when content empty AND tool calls exist
    } else {
        message["content"] = content.empty() ? nullptr : content;  // Original: use actual content otherwise
    }
    if (has_tool_calls) {
        message["tool_calls"] = tool_calls;
    }

    json choices =
        streaming ? json::array({ json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}} })
        : json::array({ json{{"finish_reason", finish_reason},
                              {"index", 0},
                              {"message", message}} });

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
    if (stopped_limit) {
        finish_reason = "length";
    } else if (stopped_word || stopped_eos) {
        // Following original llama.cpp pattern: finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls"
        // Use generated_text (complete content) for finish_reason logic, not content (empty in streaming)
        std::string generated_text = json_value(result, "generated_text", std::string(""));
        ik_chat_msg final_msg = parse_chat_message_incremental(generated_text, false, modelname);

        // Debug logging
        LOG_INFO("DEBUG: Streaming finish_reason check", {
            {"generated_text", generated_text},
            {"model_name", modelname},
            {"tool_calls_count", final_msg.tool_calls.size()}
        });

        finish_reason = final_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    std::time_t t = std::time(0);

    // Follow original llama.cpp pattern: Always process diffs and add final chunk
    std::vector<json> streaming_chunks;

    // Extract diffs from task result (populated by send_partial_response)
    // Following original llama.cpp pattern where diffs are stored in task result
    std::vector<ik_chat_msg_diff> diffs;

    if (result.contains("oaicompat_msg_diffs") && result["oaicompat_msg_diffs"].is_array()) {
        for (const auto & diff_json : result["oaicompat_msg_diffs"]) {
            ik_chat_msg_diff diff;

            // Extract content delta
            diff.content_delta = diff_json.value("content_delta", "");

            // Extract tool call data
            if (diff_json.contains("tool_call_index")) {
                diff.tool_call_index = diff_json["tool_call_index"];
                if (diff_json.contains("tool_call_delta")) {
                    const auto & tc_delta = diff_json["tool_call_delta"];
                    diff.tool_call_delta.id = tc_delta.value("id", "");
                    diff.tool_call_delta.name = tc_delta.value("name", "");
                    diff.tool_call_delta.arguments = tc_delta.value("arguments", "");
                }
            } else {
                diff.tool_call_index = std::string::npos;
            }

            diffs.push_back(diff);
        }
    }

    streaming_chunks = generate_streaming_chunks(diffs, completion_id, modelname);

    // Always add final chunk (like original llama.cpp)
    if (!finish_reason.empty()) {
        json finish_chunk = {
            {"choices", json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}}})},
            {"created", t},
            {"id", completion_id},
            {"model", modelname},
            {"object", "chat.completion.chunk"}
        };
        streaming_chunks.push_back(finish_chunk);
    }

    if (server_task_result_dict.count(task_result.id) > 0)
    {
        for (auto& chunk : streaming_chunks)
            chunk.push_back({ "timings", server_task_result_dict[task_result.id].timings.to_json() });
    }
    // Return streaming chunks (could be just final chunk if no diffs)
    if (!streaming_chunks.empty()) {
        return streaming_chunks;
    }

    // Fallback to original streaming logic for non-tool calls
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
    if (server_task_result_dict.count(task_result.id) > 0)
    {
        ret.push_back({ "timings", server_task_result_dict[task_result.id].timings.to_json() });
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


static json format_embeddings_response_oaicompat(const json& request, const json& embeddings) {
    json data = json::array();
    int i = 0;
    for (auto& elem : embeddings) {
        data.push_back(json{
            {"embedding", json_value(elem, "embedding", json::array())},
            {"index",     i++},
            {"object",    "embedding"}
            });
    }

    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", 0},
            {"total_tokens", 0}
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

    svr->set_default_headers({{"Server", "llama.cpp"}});

    // CORS preflight
    svr->Options(R"(.*)", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin",      req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods",     "POST");
        res.set_header("Access-Control-Allow-Headers",     "*");
        return res.set_content("", "application/json; charset=utf-8");
    });

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, json error_data) {
        json final_response {{"error", error_data}};
        res.set_content(final_response.dump(), "application/json; charset=utf-8");
        res.status = json_value(error_data, "code", 500);
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
        {"chat_template", ctx_server.chat_templates.template_default->source().c_str()},
    });

    LOG_INFO("chat template", {
        {"chat_example",  llama_chat_format_example(ctx_server.model, *ctx_server.chat_templates.template_default, ctx_server.params.use_jinja).c_str()},
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
        // TODO: make another middleware for CORS related logic
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WARNING("Unauthorized: Invalid API Key", {});

        return false;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key](const httplib::Request & req, httplib::Response & res) {
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
                    ctx_server.queue_tasks.post(task);

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
        ctx_server.queue_tasks.post(task);

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
        ctx_server.queue_tasks.post(task);

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

        const int id_task = ctx_server.queue_tasks.post(task);
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

        const int id_task = ctx_server.queue_tasks.post(task);
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

        const int id_task = ctx_server.queue_tasks.post(task);
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
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

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
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        json data = {
            { "system_prompt",               ctx_server.system_prompt.c_str() },
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params.n_parallel },
            { "chat_template",               ctx_server.chat_templates.template_default->source() },
        };
        if (ctx_server.params.use_jinja && ctx_server.chat_templates.template_tool_use) {
            data["chat_template_tool_use"] = ctx_server.chat_templates.template_tool_use->source();
        }
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_completions = [&ctx_server, &res_error](const httplib::Request & req, httplib::Response & res) {
        if (ctx_server.params.embedding) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        auto body = json::parse(req.body);
        const auto& chat_template = body.contains("tools") && ctx_server.chat_templates.template_tool_use ? *ctx_server.chat_templates.template_tool_use : *ctx_server.chat_templates.template_default;
        json data = oaicompat_completion_params_parse(json::parse(req.body));

        const int id_task = ctx_server.queue_tasks.get_new_id();

        ctx_server.queue_results.add_waiting_task_id(id_task);
        ctx_server.request_completion(id_task, -1, data, false, false);

        if (!json_value(data, "stream", false)) {
            server_task_result result = ctx_server.queue_results.recv(id_task);
            if (!result.error && result.stop) {
                res.set_content(result.data.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
            } else {
                res_error(res, result.data);
            }

            ctx_server.queue_results.remove_waiting_task_id(id_task);
        } else {
            const auto chunked_content_provider = [id_task, &ctx_server](size_t, httplib::DataSink & sink) {
                while (true) {
                    server_task_result result = ctx_server.queue_results.recv(id_task);
                    if (!result.error) {
                        const std::string str =
                            "data: " +
                            result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";

                        LOG_VERBOSE("data stream", {
                            { "to_send", str }
                        });

                        if (!sink.write(str.c_str(), str.size())) {
                            ctx_server.queue_results.remove_waiting_task_id(id_task);
                            return false;
                        }

                        if (result.stop) {
                            break;
                        }
                    } else {
                        const std::string str =
                            "error: " +
                            result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";

                        LOG_VERBOSE("data stream", {
                            { "to_send", str }
                        });

                        if (!sink.write(str.c_str(), str.size())) {
                            ctx_server.queue_results.remove_waiting_task_id(id_task);
                            return false;
                        }

                        break;
                    }
                }

                ctx_server.queue_results.remove_waiting_task_id(id_task);
                sink.done();

                return true;
            };

            auto on_complete = [id_task, &ctx_server] (bool) {
                // cancel
                ctx_server.request_cancel(id_task);
                ctx_server.queue_results.remove_waiting_task_id(id_task);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    const auto handle_models = [&params, &model_meta](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

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


    const auto handle_chat_completions = [&ctx_server, &params, &res_error](const httplib::Request & req, httplib::Response & res) {
        if (ctx_server.params.embedding) {
            res_error(res, format_error_response("This server does not support chat completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

        auto body = json::parse(req.body);
        const auto& chat_template = body.contains("tools") && ctx_server.chat_templates.template_tool_use ? *ctx_server.chat_templates.template_tool_use : *ctx_server.chat_templates.template_default;
        json data = oaicompat_chat_completion_params_parse(ctx_server.model,body, chat_template, params.use_jinja);


        const int id_task = ctx_server.queue_tasks.get_new_id();

        ctx_server.queue_results.add_waiting_task_id(id_task);
        ctx_server.request_completion(id_task, -1, data, false, false);

        const auto completion_id = gen_chatcmplid();
        if (!json_value(data, "stream", false)) {
            server_task_result result = ctx_server.queue_results.recv(id_task);

            if (!result.error && result.stop) {
                json result_oai = format_final_response_oaicompat(data, result.data, completion_id);

                res.set_content(result_oai.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
            } else {
                res_error(res, result.data);
            }
            ctx_server.queue_results.remove_waiting_task_id(id_task);
        } else {
            const auto chunked_content_provider = [id_task, &ctx_server, completion_id, send_done = params.send_done](size_t, httplib::DataSink & sink) {
                bool successful_completion = false;
                while (true) {
                    server_task_result result = ctx_server.queue_results.recv(id_task);
                    if (!result.error) {
                        std::vector<json> result_array = format_partial_response_oaicompat(result, completion_id);

                        for (auto it = result_array.begin(); it != result_array.end(); ++it) {
                            if (!it->empty()) {
                                const std::string str =
                                    "data: " +
                                    it->dump(-1, ' ', false, json::error_handler_t::replace) +
                                    "\n\n";
                                LOG_VERBOSE("data stream", {{"to_send", str}});
                                if (!sink.write(str.c_str(), str.size())) {
                                    ctx_server.queue_results.remove_waiting_task_id(id_task);
                                    return false;
                                }
                            }
                        }
                        if (result.stop) {
                            successful_completion = true;
                            break;
                        }
                    } else {
                        const std::string str =
                            "error: " +
                            result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";
                        LOG_VERBOSE("data stream", {{"to_send", str}});
                        if (!sink.write(str.c_str(), str.size())) {
                            ctx_server.queue_results.remove_waiting_task_id(id_task);
                            return false;
                        }
                        break;
                    }
                }
                bool ok = true;
                if (send_done && successful_completion) {
                    static const std::string done_message = "data: [DONE]\n\n";
                    LOG_VERBOSE("data stream", {{"to_send", done_message}});
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

    const auto handle_infill = [&ctx_server, &res_error](const httplib::Request & req, httplib::Response & res) {
        if (ctx_server.params.embedding) {
            res_error(res, format_error_response("This server does not support infill. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

        json data = json::parse(req.body);

        const int id_task = ctx_server.queue_tasks.get_new_id();

        ctx_server.queue_results.add_waiting_task_id(id_task);
        ctx_server.request_completion(id_task, -1, data, true, false);

        if (!json_value(data, "stream", false)) {
            server_task_result result = ctx_server.queue_results.recv(id_task);
            if (!result.error && result.stop) {
                res.set_content(result.data.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
            } else {
                res_error(res, result.data);
            }

            ctx_server.queue_results.remove_waiting_task_id(id_task);
        } else {
            const auto chunked_content_provider = [id_task, &ctx_server](size_t, httplib::DataSink & sink) {
                while (true) {
                    server_task_result result = ctx_server.queue_results.recv(id_task);
                    if (!result.error) {
                        const std::string str =
                            "data: " +
                            result.data.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";

                        LOG_VERBOSE("data stream", {
                            { "to_send", str }
                        });

                        if (!sink.write(str.c_str(), str.size())) {
                            ctx_server.queue_results.remove_waiting_task_id(id_task);
                            return false;
                        }

                        if (result.stop) {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                ctx_server.queue_results.remove_waiting_task_id(id_task);
                sink.done();

                return true;
            };

            auto on_complete = [id_task, &ctx_server] (bool) {
                ctx_server.request_cancel(id_task);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    const auto handle_tokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
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
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
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
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

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
            ctx_server.request_completion(id_task, -1, {{"prompt", prompt}}, false, true);

            // get the result
            server_task_result result = ctx_server.queue_results.recv(id_task);
            ctx_server.queue_results.remove_waiting_task_id(id_task);
            if (!result.error) {
                if (result.data.count("results")) {
                    // result for multi-task
                    responses = result.data.at("results");
                } else {
                    // result for single task
                    responses = std::vector<json>{result.data};
                }
            } else {
                // error received, ignore everything else
                res_error(res, result.data);
                return;
            }
        }

        // write JSON response
        json root = is_openai
            ? format_embeddings_response_oaicompat(body, responses)
            : responses[0];
        return res.set_content(root.dump(), "application/json; charset=utf-8");
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
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
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));

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
        const int id_task = ctx_server.queue_tasks.post(task);
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto list_saved_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
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

                response.push_back({
                    {"filename", entry.path().filename().string()},
                    {"filesize", entry.file_size()},
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

    auto handle_static_file = [](unsigned char * content, size_t len, const char * mime_type) {
        return [content, len, mime_type](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(content), len, mime_type);
            return false;
        };
    };

    //
    // Router
    //

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
            svr->Get("/", [](const httplib::Request& req, httplib::Response& res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                }
                else {
                    res.set_header("Content-Encoding", "gzip");
                    // COEP and COOP headers, required by pyodide (python interpreter)
                    res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                    res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                }
                return false;
                });
        }
    }
    // register API routes
    svr->Get ("/health",              handle_health);
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Get ("/v1/models",           handle_models);
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions",         handle_completions);
    svr->Post("/v1/completions",      handle_completions);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    if (!params.slot_save_path.empty()) {
        // these endpoints rely on slot_save_path existing
        svr->Post("/slots/:id_slot",  handle_slots_action);
        svr->Get ("/list",            list_saved_prompts);
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

    ctx_server.queue_tasks.on_new_task(std::bind(
        &server_context::process_single_task, &ctx_server, std::placeholders::_1));
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
