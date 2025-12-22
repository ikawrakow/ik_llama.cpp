#include "server-task.h"


json result_timings::to_json() const {
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


json server_task_result::to_json_final() {
    switch (oaicompat) {
    case OAICOMPAT_TYPE_NONE:
        return to_json_non_oaicompat_final();
    case OAICOMPAT_TYPE_COMPLETION:
        return to_json_oaicompat_final();
    case OAICOMPAT_TYPE_CHAT:
        return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat_final();
    case OAICOMPAT_TYPE_ANTHROPIC:
        return stream ? to_json_anthropic_stream() : to_json_anthropic_final();
    default:
        GGML_ASSERT(false && "Invalid oaicompat_type");
    }
}

json server_task_result::to_json_partial() {
    switch (oaicompat) {
    case OAICOMPAT_TYPE_NONE:
        return to_json_non_oaicompat_partial();
    case OAICOMPAT_TYPE_COMPLETION:
        return to_json_oaicompat_partial();
    case OAICOMPAT_TYPE_CHAT:
        return to_json_oaicompat_chat_partial();
    case OAICOMPAT_TYPE_ANTHROPIC:
        return to_json_anthropic_partial();
    default:
        GGML_ASSERT(false && "Invalid oaicompat_type");
    }
}

json server_task_result::to_json_non_oaicompat_partial() {
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

json server_task_result::to_json_non_oaicompat_final() {
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

json server_task_result::to_json_oaicompat_partial() {
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

json server_task_result::to_json_oaicompat_final() {
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

json server_task_result::to_json_oaicompat_chat_partial() {
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

json server_task_result::to_json_oaicompat_chat_final() {
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

json server_task_result::to_json_oaicompat_chat_stream() {
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

json server_task_result::to_json_anthropic_final() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }


    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
            });
    }

    for (const auto& tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        }
        catch (const std::exception&) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"input_tokens", n_prompt_tokens},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_text = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    bool text_block_started = false;
    std::set<size_t> tool_calls_started;

    for (const auto& diff : oaicompat_msg_diffs) {

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", 0},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", 0},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto& full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                    });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", 0}
            }}
            });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
            });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
        });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
        });

    // extra fields for debugging purposes
    if (verbose && !events.empty()) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_final();
    }
    // Don't add timings for Anthropic API (breaks spec compliance)
    if (oaicompat != OAICOMPAT_TYPE_ANTHROPIC && timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    return events;
}

json server_task_result::to_json_anthropic_partial() {
    json events = json::array();
    bool first = n_decoded == 1;
    static bool text_block_started = false;

    if (first) {
        text_block_started = false;

        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"input_tokens", n_prompt_tokens},
                        {"output_tokens", 0}
                    }}
                }}
            }}
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", 0},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", 0},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (text_block_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                    });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (verbose && !events.empty() && first) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_partial();
    }

    if (timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    //if (is_progress && !events.empty()) {
    //    events.back()["data"]["prompt_progress"] = progress.to_json();
    //}

    return events;
}


size_t server_prompt::size() const {
    size_t res = data.size();

    for (const auto& checkpoint : checkpoints) {
        res += checkpoint.size();
    }

    return res;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.n_tokens();
    }
    return res;

}

bool server_prompt_cache::load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot) {
    thinking_tokens think_tokens;
    for (auto it = states.begin(); it != states.end(); ++it) {
        think_tokens = it->think_tokens;
        break;
    }
    server_tokens prompt_tokens;
    server_tokens tokens_new_ex;
    if (think_tokens.exclude) {
        prompt_tokens = server_tokens(prompt.tokens.get_text_tokens_exclude_think(ctx, think_tokens), false);
        tokens_new_ex = server_tokens(tokens_new.get_text_tokens_exclude_think(ctx, think_tokens), false);
    }
    else {
        prompt_tokens = std::move(prompt.tokens); //server_tokens(prompt.tokens.get_text_tokens(), false);
        tokens_new_ex = server_tokens(tokens_new.get_text_tokens(), false);
    }
    const auto lcp_best = prompt_tokens.get_common_prefix(ctx, tokens_new_ex);
    float f_keep_best = float(lcp_best.second) / prompt_tokens.size();
    float sim_best = prompt_tokens.get_tokens_similarity(ctx, tokens_new_ex, prompt.n_kept_prompt, prompt.n_discarded_prompt);
    LLAMA_LOG_INFO(" - looking for better prompt, base f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, prompt.n_kept_prompt, prompt.n_discarded_prompt);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        server_tokens tokens;
        if (think_tokens.exclude) {
            tokens = server_tokens(it->tokens.get_text_tokens_exclude_think(ctx, think_tokens), false);
        }
        else {
            tokens = std::move(it->tokens);
        }
        const auto lcp_cur = tokens.get_common_prefix(ctx, tokens_new_ex);
        const float f_keep_cur = float(lcp_cur.first) / tokens.size();
        const float sim_cur = tokens.get_tokens_similarity(ctx, tokens_new_ex, it->n_kept_prompt, it->n_discarded_prompt);
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

server_prompt* server_prompt_cache::alloc(const server_prompt& prompt, size_t state_size) {
    for (auto it = states.begin(); it != states.end();) {
        auto tokens_ctx_shift = server_tokens(prompt.tokens.get_text_tokens(), false); // copy cache tokens
        tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);
        auto prefix = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
        const size_t len = prefix.first;
        const size_t len_prompt = prefix.second;
        // first check if the current state is contained fully in the cache
        if (len_prompt == tokens_ctx_shift.size()) {
            LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
        // next, remove any cached prompts that are fully contained in the current prompt
        else if (len == it->tokens.size()) {
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
        /*.think_tokens                   =*/ prompt.think_tokens,
        /*.data            =*/ std::move(state_data),
        /*.checkpoints     =*/ prompt.checkpoints,
    };

    return &cur;
}


void server_prompt_cache::update() {
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

    LLAMA_LOG_INFO(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
        states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto& state : states) {
        LLAMA_LOG_INFO("   - prompt %p: %7d tokens, %7d discarded, checkpoints: %2zu, %9.3f MiB\n",
            (const void*)&state, state.n_tokens(), state.n_discarded_prompt, state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
