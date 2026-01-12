#include "server-context.h"
#include "server-common.h"
#include "server-task.h"
#include "server-queue.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"


server_context::~server_context() {
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
    for (server_slot& slot : slots) {
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

bool server_context::load_model(const gpt_params& params_) {
    params = params_;

    llama_init_result llama_init = llama_init_from_gpt_params(params);

    model = llama_init.model;
    ctx = llama_init.context;
    lora_adapters = llama_init.lora_adapters;

    if (model == nullptr) {
        LOG_ERROR("unable to load model", { {"model", params.model} });
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
    std::string& mmproj_path = params.mmproj.path;
    if (!mmproj_path.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.print_timings = false;
        mparams.n_threads = params.n_threads;
        mparams.flash_attn_type = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        mparams.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
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
        params_dft.devices      = params.devices_draft;
        params_dft.model        = params.model_draft;
        params_dft.n_gpu_layers = params.n_gpu_layers_draft;
        params_dft.rpc_servers  = params.rpc_servers;
        params_dft.cache_type_k = params.cache_type_k_draft.empty() ? params.cache_type_k : params.cache_type_k_draft;
        params_dft.cache_type_v = params.cache_type_v_draft.empty() ? params.cache_type_v : params.cache_type_v_draft;
        params_dft.flash_attn   = params.flash_attn;
        if (!params.draft_params.empty()) {
            auto [argc, argv] = parse_command_line("llama-server " + params.draft_params);
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
        params_dft.n_batch = params_dft.n_ctx;
        llama_init_result llama_init_dft = llama_init_from_gpt_params(params_dft);

        llama_model* model_dft = llama_init_dft.model;
        if (model_dft == nullptr) {
            LOG_ERROR("failed to load draft model", { {"model", params.model_draft} });
            return false;
        }

        if (!llama_speculative_are_compatible(ctx, llama_init_dft.context)) {
            LOG_INFO("the draft model is not compatible with the target model. tokens will be translated between the draft and target models.", { {} });
        }

        const int n_ctx_dft = llama_n_ctx(llama_init_dft.context);

        cparams_dft = llama_context_params_from_gpt_params(params_dft);

        model_draft = llama_init_dft.model;
        ctx_draft = llama_init_dft.context;
    }
    return true;
}

void server_context::init() {
    const int32_t n_ctx_slot = n_ctx / params.n_parallel;

    LOG_INFO("initializing slots", { {"n_slots", params.n_parallel} });

    for (int i = 0; i < params.n_parallel; i++) {
        server_slot slot;

        slot.id = i;
        slot.ctx = ctx;
        slot.n_ctx = n_ctx_slot;
        slot.n_predict = params.n_predict;
        slot.mctx = mctx;
        slot.cache_tokens.has_mtmd = mctx != nullptr;
        slot.params.think_tokens = params.think_tokens;
        if (params.think_tokens.exclude) {
            SRV_WRN("Exclude reasoning tokens when selecting slot based on similarity: start: %s, end: %s\nuse `--reasoning-tokens none` to disable.\n", params.think_tokens.begin.c_str(), params.think_tokens.end.c_str() );
        }
        else {
            SRV_WRN("%s", "Include reasoning tokens when selecting slot based on similarity\nuse `--reasoning-tokens auto` to exclude reasoning tokens.\n");
        }
        LOG_INFO("new slot", {
            {"id_slot",    slot.id},
            {"n_ctx_slot", slot.n_ctx}
            });

        const int ga_n = params.grp_attn_n;
        const int ga_w = params.grp_attn_w;

        if (ga_n != 1) {
            GGML_ASSERT(ga_n > 0 && "ga_n must be positive");                       // NOLINT
            GGML_ASSERT(ga_w % ga_n == 0 && "ga_w must be a multiple of ga_n");             // NOLINT
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
            for (auto& pair : params.replacements_draft) {
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
        prompt_cache = std::make_unique<server_prompt_cache>(ctx, params.cache_ram_mib, 0);
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


void server_slot::prompt_save(server_prompt_cache& prompt_cache) const {
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

void server_slot::prompt_load(server_prompt_cache& prompt_cache, const server_tokens& tokens) {
    bool res = prompt_cache.load(server_cached_prompt, tokens, ctx, id);
    if (!res) {
        LLAMA_LOG_INFO("failed to load prompt from cache\n");
    }
}

void server_slot::reset() {
    n_prompt_tokens = 0;
    generated_text = "";
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    n_past = 0;
    n_sent_text = 0;

    drafted.clear();
    i_batch_dft.clear();

    n_sent_token_probs = 0;
    infill = false;
    ga_i = 0;
    n_past_se = 0;
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

bool server_slot::has_budget(gpt_params& global_params) {
    if (params.n_predict == -1 && global_params.n_predict == -1) {
        return true; // limitless
    }

    n_remaining = -1;

    if (params.n_predict != -1) {
        n_remaining = params.n_predict - n_decoded;
    }
    else if (global_params.n_predict != -1) {
        n_remaining = global_params.n_predict - n_decoded;
    }

    return n_remaining > 0; // no budget
}

bool server_slot::available() const {
    return state == SLOT_STATE_IDLE && command == SLOT_COMMAND_NONE;
}

bool server_slot::is_processing() const {
    return (state == SLOT_STATE_IDLE && command == SLOT_COMMAND_LOAD_PROMPT) || state == SLOT_STATE_PROCESSING;
}

void server_slot::add_token_string(const completion_token_output& token) {
    if (command == SLOT_COMMAND_RELEASE) {
        return;
    }
    generated_token_probs.push_back(token);
}

int server_slot::get_n_draft_max() const {
    if (!ctx_dft) {
        return 0;
    }

    // determine the max draft that fits the current slot state
    int n_draft_max = params.speculative.n_max;

    // note: slot.prompt is not yet expanded with the `id` token sampled above
    //       also, need to leave space for 1 extra token to allow context shifts
    n_draft_max = std::min(n_draft_max, n_ctx - n_past - 2);

    if (n_remaining > 0) {
        n_draft_max = std::min(n_draft_max, n_remaining - 1);
    }

    SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

    if (n_draft_max < params.speculative.n_min) {
        SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, params.speculative.n_min);
        n_draft_max = 0;
    }
    return n_draft_max;
}

void server_slot::release() {
    if (state == SLOT_STATE_PROCESSING) {
        t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
        command = SLOT_COMMAND_RELEASE;
        task.reset();
    }
}


json server_slot::get_formated_timings() const {
    return json{
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

result_timings server_slot::get_timings() const {
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

const common_chat_msg& server_slot::update_chat_msg(std::vector<common_chat_msg_diff>& diffs) {
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


size_t server_slot::find_stopping_strings(const std::string& text, const size_t last_token_size, bool is_full_stop) {
    size_t stop_pos = std::string::npos;

    for (const std::string& word : params.antiprompt) {
        size_t pos;

        if (is_full_stop) {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

            pos = text.find(word, from_pos);
        }
        else {
            pos = string_find_partial_stop(text, word);
        }

        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            if (is_full_stop) {
                stopped_word = true;
                stopping_word = word;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }

    return stop_pos;
}

void server_slot::print_timings() const {
    char buffer[512];
    double t_token = t_prompt_processing / n_prompt_tokens_processed;
    double n_tokens_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

    //snprintf(buffer, 512, "prompt eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)",
    //    t_prompt_processing, n_prompt_tokens_processed,
    //    t_token, n_tokens_second);

    //LOG_INFO(buffer, {});

    double t_token_gen = t_token_generation / n_decoded;
    double n_tokens_second_gen = 1e3 / t_token_generation * n_decoded;

    //snprintf(buffer, 512, "generation eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)",
    //    t_token_generation, n_decoded,
    //    t_token, n_tokens_second);

    //LOG_INFO(buffer, {});

    //snprintf(buffer, 512, "          total time = %10.2f ms", t_prompt_processing + t_token_generation);

    //LOG_INFO(buffer, {});
    SLT_INF(*this,
        "\n"
        "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
        "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
        "      total time = %10.2f ms / %5d tokens\n",
        t_prompt_processing, n_prompt_tokens_processed, t_token, n_tokens_second,
        t_token_generation, n_decoded, t_token_gen, n_tokens_second_gen,
        t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

    if (n_draft_total > 0) {
        const float draft_ratio = (float)n_draft_accepted / n_draft_total;
        SLT_CNT(*this,
            "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
            draft_ratio, n_draft_accepted, n_draft_total
        );
    }
}

void server_metrics::init() {
    t_start = ggml_time_us();
}

void server_metrics::on_prompt_eval(const server_slot& slot) {
    n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
    n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
    t_prompt_processing += slot.t_prompt_processing;
    t_prompt_processing_total += slot.t_prompt_processing;
}

void server_metrics::on_prediction(const server_slot& slot) {
    n_tokens_predicted_total += slot.n_decoded;
    n_tokens_predicted += slot.n_decoded;
    t_tokens_generation += slot.t_token_generation;
    t_tokens_generation_total += slot.t_token_generation;
}

void server_metrics::reset_bucket() {
    n_prompt_tokens_processed = 0;
    t_prompt_processing = 0;
    n_tokens_predicted = 0;
    t_tokens_generation = 0;
}

std::vector<llama_token> server_context::tokenize(const json& json_prompt, bool add_special) const {
    // TODO: currently, we tokenize using special tokens by default
    //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
    //       but it's better compared to completely ignoring ChatML and other chat templates
    const bool TMP_FORCE_SPECIAL = true;

    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    std::vector<llama_token> prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto& p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                std::vector<llama_token> p;
                if (first) {
                    p = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
                    first = false;
                }
                else {
                    p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            }
            else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    }
    else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
    }

    return prompt_tokens;
}

server_slot* server_context::get_slot_by_id(int id) {
    for (server_slot& slot : slots) {
        if (slot.id == id) {
            return &slot;
        }
    }

    return nullptr;
}

float server_context::calculate_slot_f_keep(const server_slot & slot, llama_context * ctx,const server_tokens & a, const server_tokens & b) {
    float f_keep = 0.0f;
    if (!a.empty()) {
        if (slot.ga_n == 1 && slot.n_discarded_prompt > 0 && b.size() >= slot.n_ctx) {
            f_keep = a.get_cached_tokens_similarity(slot.ctx, b, slot.params.n_keep + add_bos_token, slot.n_discarded_prompt);
        }
        else {
            f_keep = a.get_cached_tokens_similarity(slot.ctx, b, 0, 0);
        }
    }
    return f_keep;
}

std::pair<common_prefix, float> server_context::calculate_slot_similarity(const server_slot& slot, llama_context* ctx, const server_tokens& a, const server_tokens& b) {
    std::pair<common_prefix, float> sim;
    // length of the Longest Common Prefix between the current slot's prompt and the input prompt
    common_prefix lcp_len = a.get_common_prefix(slot.ctx, b);
    // fraction of the Longest Common Prefix length with respect to the input prompt and cached prompt length
    float sim_cur = a.get_tokens_similarity(slot.ctx, b, 0, 0);
    // handle context shift
    if (slot.ga_n == 1 && slot.n_discarded_prompt > 0 && b.size() >= slot.n_ctx) {
        float sim_cur_ctx_shift = a.get_tokens_similarity(slot.ctx, b, slot.n_kept_prompt, slot.n_discarded_prompt);
        if (sim_cur_ctx_shift > sim_cur) {
            sim_cur = sim_cur_ctx_shift;
        }
    }
    sim.first = lcp_len;
    sim.second = sim_cur;
    return sim;
}

void server_context::copy_data_to_cached_prompt(const server_tokens & tokens, server_slot & slot) {
    slot.server_cached_prompt.tokens = server_tokens(tokens.get_text_tokens(), false); // copy cache tokens
    slot.server_cached_prompt.n_discarded_prompt = slot.n_discarded_prompt;
    slot.server_cached_prompt.n_kept_prompt = slot.n_kept_prompt;
    slot.server_cached_prompt.think_tokens = slot.params.think_tokens;
}

server_slot* server_context::get_available_slot(const server_task& task) {
    server_slot* ret = nullptr;
    bool update_cache = false;

    // find the slot that has at least n% prompt similarity
    if (ret == nullptr && slot_prompt_similarity != 0.0f) {
        int max_lcp_len = 0;
        float sim_best = 0;

        for (server_slot& slot : slots) {
            // skip the slot if it is not available
            if (!slot.available()) {
                continue;
            }
            auto& cache_tokens = slot.cache_tokens;
            // skip the slot if it does not contains prompt
            if (cache_tokens.empty()) {
                continue;
            }
            bool exclude_think = !cache_tokens.has_mtmd && slot.params.think_tokens.exclude;
            std::pair<common_prefix, float> sim;
            if (exclude_think) {
                auto temp = slot.cache_tokens.get_text_tokens_exclude_think(slot.ctx, slot.params.think_tokens);
                server_tokens cache_tokens_exclude_think = server_tokens(temp, false);
                temp = task.tokens.get_text_tokens_exclude_think(slot.ctx, slot.params.think_tokens);
                server_tokens prompt_tokens_exclude_think = server_tokens(temp, false);
                sim = calculate_slot_similarity(slot, ctx, cache_tokens_exclude_think, prompt_tokens_exclude_think);
            }
            else {
                sim = calculate_slot_similarity(slot, ctx, cache_tokens, task.tokens);
            }
            common_prefix lcp_len = sim.first;
            float sim_cur = sim.second;

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
        for (server_slot& slot : slots) {
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
        auto& tokens = ret->cache_tokens;
        float f_keep = 0;
        size_t cache_token_size = tokens.size();
        if (!tokens.empty()) {
            bool exclude_think = !tokens.has_mtmd && ret->params.think_tokens.exclude;
            if (exclude_think) {
                auto temp = tokens.get_text_tokens_exclude_think(ret->ctx, ret->params.think_tokens);
                server_tokens cache_exclude_think = server_tokens(temp, false);

                temp = task.tokens.get_text_tokens_exclude_think(ret->ctx, ret->params.think_tokens);
                server_tokens prompt_exclude_think = server_tokens(temp, false);

                cache_token_size = cache_exclude_think.size();
                f_keep = calculate_slot_f_keep(*ret, ret->ctx, cache_exclude_think, prompt_exclude_think);
            }
            else {
                f_keep = calculate_slot_f_keep(*ret, ret->ctx, tokens, task.tokens);
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
        update_cache = update_cache && cache_token_size >= cache_ram_n_min;

        // TODO: mtmd does not support prompt cache
        update_cache = update_cache && (ret->mctx == nullptr);

        LLAMA_LOG_INFO("======== Prompt cache: cache size: %d, n_keep: %d, n_discarded_prompt: %d, cache_ram_n_min: %d, f_keep: %.2f, cache_ram_similarity: %.2f\n",
            (int)tokens.size(), ret->n_kept_prompt, ret->n_discarded_prompt, cache_ram_n_min, f_keep, cache_ram_similarity);
        if (update_cache) {
            const int64_t t_start = ggml_time_us();
            LLAMA_LOG_INFO("updating prompt cache\n");
            // copy cache tokens
            copy_data_to_cached_prompt(tokens, *ret);

            ret->prompt_save(*prompt_cache);
            LLAMA_LOG_INFO("prompt cache save took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
        }
        // has prompts saved earlier to load
        if (prompt_cache && !prompt_cache->states.empty()) {
            const int64_t t_start = ggml_time_us();
            copy_data_to_cached_prompt(tokens, *ret);

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

bool server_context::launch_slot_with_task(server_slot& slot, server_task& task) {
    slot_params default_params;
    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    llama_sampling_params default_sparams = params.sparams;
    auto& data = task.data;

    if (data.count("__oaicompat") != 0) {
        slot.oaicompat = true;
        slot.oaicompat_model = json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
    }
    else {
        slot.oaicompat = false;
        slot.oaicompat_model = "";
    }
    slot.params.timings_per_token = json_value(data, "timings_per_token", false);
    slot.params.stream = json_value(data, "stream", false);
    auto stream_opt = json_value(data, "stream_options", json::object());
    slot.params.include_usage = json_value(stream_opt, "include_usage", false);
    slot.params.cache_prompt = json_value(data, "cache_prompt", true);
    slot.params.n_predict = json_value(data, "n_predict", json_value(data, "max_tokens", default_params.n_predict));
    slot.sparams.top_k = json_value(data, "top_k", default_sparams.top_k);
    slot.sparams.top_p = json_value(data, "top_p", default_sparams.top_p);
    slot.sparams.min_p = json_value(data, "min_p", default_sparams.min_p);
    slot.sparams.tfs_z = json_value(data, "tfs_z", default_sparams.tfs_z);
    slot.sparams.typical_p = json_value(data, "typical_p", default_sparams.typical_p);
    slot.sparams.temp = json_value(data, "temperature", default_sparams.temp);
    slot.sparams.dynatemp_range = json_value(data, "dynatemp_range", default_sparams.dynatemp_range);
    slot.sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
    slot.sparams.xtc_probability = json_value(data, "xtc_probability", default_sparams.xtc_probability);
    slot.sparams.xtc_threshold = json_value(data, "xtc_threshold", default_sparams.xtc_threshold);
    slot.sparams.top_n_sigma = json_value(data, "top_n_sigma", default_sparams.top_n_sigma);
    slot.sparams.penalty_last_n = json_value(data, "repeat_last_n", default_sparams.penalty_last_n);
    slot.sparams.penalty_repeat = json_value(data, "repeat_penalty", default_sparams.penalty_repeat);
    slot.sparams.penalty_freq = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
    slot.sparams.penalty_present = json_value(data, "presence_penalty", default_sparams.penalty_present);
    slot.sparams.dry_multiplier = json_value(data, "dry_multiplier", default_sparams.dry_multiplier);
    slot.sparams.dry_base = json_value(data, "dry_base", default_sparams.dry_base);
    slot.sparams.dry_allowed_length = json_value(data, "dry_allowed_length", default_sparams.dry_allowed_length);
    slot.sparams.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", default_sparams.dry_penalty_last_n);
    slot.sparams.mirostat = json_value(data, "mirostat", default_sparams.mirostat);
    slot.sparams.mirostat_tau = json_value(data, "mirostat_tau", default_sparams.mirostat_tau);
    slot.sparams.mirostat_eta = json_value(data, "mirostat_eta", default_sparams.mirostat_eta);
    slot.sparams.adaptive_target = json_value(data, "adaptive_target", default_sparams.adaptive_target);
    slot.sparams.adaptive_decay = json_value(data, "adaptive_decay", default_sparams.adaptive_decay);
    slot.sparams.penalize_nl = json_value(data, "penalize_nl", default_sparams.penalize_nl);
    slot.params.n_keep = json_value(data, "n_keep", slot.params.n_keep);
    slot.params.n_discard = json_value(data, "n_discard", default_params.n_discard);
    slot.sparams.seed = json_value(data, "seed", default_sparams.seed);
    slot.sparams.n_probs = json_value(data, "n_probs", default_sparams.n_probs);
    slot.sparams.min_keep = json_value(data, "min_keep", default_sparams.min_keep);

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
            auto schema = json_value(data, "json_schema", json::object());
            LLAMA_LOG_DEBUG("JSON schema: %s\n", schema.dump(2).c_str());
            slot.sparams.grammar = json_schema_to_grammar(schema);
            LLAMA_LOG_DEBUG("Converted grammar: %s\n", slot.sparams.grammar.c_str());
        }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    }
    else {
        slot.sparams.grammar = json_value(data, "grammar", default_sparams.grammar);
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
        const auto& prompt = data.find("prompt");
        if (!slot.prompt_tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (prompt == data.end()) {
            send_error(task, "\"prompt\" must be provided", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        if ((prompt->is_string()) ||
            (prompt->is_array() && prompt->size() == 1 && prompt->at(0).is_string()) ||
            (prompt->is_array() && !prompt->empty() && prompt->at(0).is_number_integer())) {
            slot.prompt = *prompt;
        }
        else if (prompt->is_array() && prompt->size() == 1 && prompt->at(0).is_array()) {
            slot.prompt = prompt->at(0);
        }
        else {
            send_error(task, "\"prompt\" must be a string or an array of integers", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        slot.prompt_tokens = std::move(task.tokens);
    }

    // penalize user-provided tokens
    {
        slot.sparams.penalty_prompt_tokens.clear();
        slot.sparams.use_penalty_prompt_tokens = false;

        const auto& penalty_prompt = data.find("penalty_prompt");

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
                for (const auto& penalty_token : *penalty_prompt) {
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

        const auto& logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_n_vocab(model);
            for (const auto& el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    }
                    else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    }
                    else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            slot.sparams.logit_bias[tok] = bias;
                        }
                    }
                    else if (el[0].is_string()) {
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

        const auto& stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            for (const auto& word : *stop) {
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
        slot.ctx_sampling = llama_sampling_init(llama_get_model_vocab(model), slot.sparams);
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

void server_context::kv_cache_clear() {
    LOG_VERBOSE("clearing KV cache", {});

    // clear the entire KV cache
    llama_kv_cache_clear(ctx);
    clean_kv_cache = false;
}

void server_context::system_prompt_update() {
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

bool server_context::system_prompt_set(const std::string& sys_prompt) {
    system_prompt = sys_prompt;

    LOG_VERBOSE("system prompt process", {
        {"system_prompt",  system_prompt},
        });

    // release all slots
    for (server_slot& slot : slots) {
        slot.release();
    }

    system_need_update = true;
    return true;
}

bool server_context::process_token(completion_token_output& result, server_slot& slot) {
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
        }
        else {
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
        slot.stopped_limit = true;
        slot.has_next_token = false;

        LOG_VERBOSE("stopped by limit", {
            {"id_slot",   slot.id},
            {"id_task",   slot.id_task},
            {"n_decoded", slot.n_decoded},
            {"n_predict", slot.params.n_predict},
            });
    }

    if (llama_token_is_eog(model, result.tok)) {
        slot.stopped_eos = true;
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
        slot.truncated = true;
        slot.stopped_limit = true;
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

void server_context::populate_token_probs(const server_slot& slot, completion_token_output& result, bool post_sampling, bool special, int idx) {
    size_t n_probs = slot.sparams.n_probs;
    size_t n_vocab = llama_n_vocab(llama_get_model(ctx));

    if (post_sampling) {
        const auto* cur_p = llama_sampling_get_candidates(slot.ctx_sampling);
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
    }
    else {
        auto&& [sampled_token_p, cur] = get_token_probabilities(ctx, idx, result.tok, n_probs);

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

json server_context::get_formated_generation(const server_slot& slot) const {
    const auto eos_bias = slot.sparams.logit_bias.find(llama_token_eos(model));
    const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() && eos_bias->second < 0.0f && std::isinf(eos_bias->second);

    std::vector<std::string> samplers_sequence;
    samplers_sequence.reserve(slot.sparams.samplers_sequence.size());
    for (const auto& sampler_type : slot.sparams.samplers_sequence) {
        samplers_sequence.emplace_back(llama_sampling_type_to_str(sampler_type));
    }

    auto grammar_triggers = json::array();
    for (const auto& trigger : slot.sparams.grammar_triggers) {
        grammar_triggers.push_back(trigger.to_json<json>());
    }

    return json{
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
        {"adaptive_target",           slot.sparams.adaptive_target},
        {"adaptive_decay",            slot.sparams.adaptive_decay},
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

void server_context::send_error(const server_task& task, const std::string& error, const enum error_type type) {
    send_error(task.id, task.id_multi, error, type);
}

void server_context::send_error(const server_slot& slot, const std::string& error, const enum error_type type) {
    send_error(slot.id_task, slot.id_multi, error, type);
}

void server_context::send_error(const int id_task, const int id_multi, const std::string& error, const enum error_type type ) {
    LOG_ERROR("task error", {
        {"id_multi", id_multi},
        {"id_task", id_task},
        {"error", error},
        });

    server_task_result res;
    res.id = id_task;
    res.id_multi = id_multi;
    res.stop = false;
    res.error = true;
    res.data = format_error_response(error, type);

    queue_results.send(res);
}

// if multimodal is enabled, send an error and return false
bool server_context::ensure_no_mtmd(const int id_task) {
    if (mctx) {
        int id_multi = 0;
        send_error(id_task, id_multi, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
        return false;
    }
    return true;
}

void server_context::send_partial_response(server_slot& slot, completion_token_output tkn) {
    server_task_result res;
    res.final_result = false;
    res.id = slot.id_task;
    res.id_multi = slot.id_multi;
    res.error = false;
    res.stop = false;
    res.stream = slot.params.stream;
    res.content = tkn.text_to_send;
    res.post_sampling_probs = slot.params.post_sampling_probs;
    res.oaicompat = slot.params.oaicompat;
    res.oaicompat_model = slot.params.oaicompat_model;
    res.oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;
    res.n_decoded = slot.n_decoded;
    res.n_prompt_tokens = slot.n_prompt_tokens;
    res.data = json{
        {"content",    tkn.text_to_send},
        {"stop",       false},
        {"id_slot",    slot.id},
        {"multimodal", false}
    };
    slot.update_chat_msg(res.oaicompat_msg_diffs);

    // populate res.probs_output
    if (slot.sparams.n_probs > 0) {
        res.probs_output = { tkn }; // copy the token probs
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

void server_context::send_final_response(server_slot& slot) {
    server_task_result res;
    res.final_result = true;
    res.id = slot.id_task;
    res.id_multi = slot.id_multi;
    res.error = false;
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
    res.data = json{
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

void server_context::send_embedding(const server_slot& slot, const llama_batch& batch) {
    server_task_result res;
    res.id = slot.id_task;
    res.id_multi = slot.id_multi;
    res.error = false;
    res.stop = true;

    const int n_embd = llama_n_embd(model);

    std::vector<float> embd_res(n_embd, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
            continue;
        }

        const float* embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
        }

        if (embd == NULL) {
            LOG_ERROR("failed to get embeddings", {
                {"token",  batch.token[i]},
                    {"seq_id", batch.seq_id[i][0]}
                });

            res.data = json{
                {"embedding", std::vector<float>(n_embd, 0.0f)},
                {"tokens_evaluated", slot.n_prompt_tokens},
            };

            continue;
        }

        llama_embd_normalize(embd, embd_res.data(), n_embd);

        res.data = json{
            {"embedding", embd_res},
            {"tokens_evaluated", slot.n_prompt_tokens},
        };
    }

    queue_results.send(res);
}

void server_context::request_completion(int id_task, int id_multi, json data, bool infill, bool embedding, server_tokens&& inputs) {
    server_task task;
    task.id = id_task;
    task.id_multi = id_multi;
    task.id_target = 0;
    task.data = std::move(data);
    task.infill = infill;
    task.embedding = embedding;
    task.type = SERVER_TASK_TYPE_COMPLETION;
    task.tokens = std::move(inputs);
    // when a completion task's prompt array is not a singleton, we split it into multiple requests
    // otherwise, it's a single-prompt task, we actually queue it
    // if there's numbers in the prompt array it will be treated as an array of tokens
    if (task.data.count("prompt") != 0 && task.data.at("prompt").size() > 1) {
        bool numbers = false;
        for (const auto& e : task.data.at("prompt")) {
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
        }
        else {
            split_multiprompt_task(id_task, task);
        }
    }
    else {
        queue_tasks.post(std::move(task));
    }
}

void server_context::request_cancel(int id_task) {
    server_task task;
    task.type = SERVER_TASK_TYPE_CANCEL;
    task.id_target = id_task;

    queue_tasks.post(std::move(task));
}

void server_context::split_multiprompt_task(int id_multi, server_task& multiprompt_task) {
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

void server_context::process_single_task(server_task&& task) {
    switch (task.type) {
    case SERVER_TASK_TYPE_COMPLETION:
    {
        const int id_slot = json_value(task.data, "id_slot", -1);

        server_slot* slot;

        if (id_slot != -1) {
            slot = get_slot_by_id(id_slot);
        }
        else {
            slot = get_available_slot(task);
        }

        if (slot == nullptr) {
            // if no slot is available, we defer this task for processing later
            LOG_VERBOSE("no slot is available", { {"id_task", task.id} });
            queue_tasks.defer(std::move(task));
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
            queue_tasks.defer(std::move(task));
            break;
        }

        if (task.data.contains("system_prompt")) {
            std::string sys_prompt = json_value(task.data, "system_prompt", std::string());
            system_prompt_set(sys_prompt);

            for (server_slot& slot : slots) {
                slot.n_past = 0;
                slot.n_past_se = 0;
            }
        }

        slot->reset();

        slot->id_task = task.id;
        slot->id_multi = task.id_multi;
        slot->infill = task.infill;
        slot->embedding = task.embedding;

        if (!launch_slot_with_task(*slot, task)) {
            LOG_ERROR("error while launching slot", task.data);
            break;
        }
    } break;
    case SERVER_TASK_TYPE_CANCEL:
    {
        // release slot linked with the task id
        for (auto& slot : slots) {
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

        int n_idle_slots = 0;
        int n_processing_slots = 0;

        for (server_slot& slot : slots) {
            json slot_data = get_formated_generation(slot);
            slot_data["id"] = slot.id;
            slot_data["id_task"] = slot.id_task;
            slot_data["state"] = slot.state;
            slot_data["prompt"] = slot.prompt;
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
            }
            else {
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
        res.id = task.id;
        res.id_multi = task.id_multi;
        res.stop = true;
        res.error = false;
        res.data = {
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
        server_slot* slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
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
        result.data = json{
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
        server_slot* slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
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
        result.data = json{
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
        server_slot* slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            break;
        }
        if (!slot->available()) {
            // if requested slot is unavailable, we defer this task for processing later
            LOG_VERBOSE("requested slot is unavailable", { {"id_task", task.id} });
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
        result.data = json{
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
        result.data = json{ { "success", true } };
        queue_results.send(result);
    } break;
    }
}

void server_context::on_finish_multitask(const server_task_multi& multitask) {
    // all subtasks done == multitask is done
    server_task_result result;
    result.id = multitask.id;
    result.stop = true;
    result.error = false;

    // collect json results into one json result
    std::vector<json> result_jsons;
    for (const auto& subres : multitask.results) {
        result_jsons.push_back(subres.data);
        result.error = result.error && subres.error;
    }
    result.data = json{
        { "results", result_jsons }
    };

    queue_results.send(result);
}

void server_context::print_tokens(const server_tokens& prompt, const server_tokens& cache, size_t start1, size_t start2, size_t length) {
    if (cache.size() > start2) {
        LLAMA_LOG_INFO("cache : %s\n", cache.detokenize(ctx, true, start2, length).c_str());
    }
    if (prompt.size() > start1) {
        LLAMA_LOG_INFO("prompt: %s\n", prompt.detokenize(ctx, true, start1, length).c_str());
    }

}

void server_context::discard_n_kv_and_cache_tokens(llama_context* ctx, server_slot& slot, int32_t n_keep, int32_t n_discard) {
    llama_kv_cache_seq_rm(ctx, slot.id, n_keep, n_keep + n_discard);
    llama_kv_cache_seq_add(ctx, slot.id, n_keep + n_discard, system_tokens.size() + slot.n_past, -n_discard);
    if (slot.params.cache_prompt) {
        slot.cache_tokens.discard_n_tokens(n_keep, n_discard);
    }
}

// convert keep first few and discard next tokens in a to b
void server_context::context_shift_find_n_tokens(llama_context* ctx, const server_tokens& a, const server_tokens& b, int32_t n_keep,
    int32_t n_discard, int32_t& n_kept, int32_t& n_discarded, bool exact) {

    common_prefix ctx_keep_prefix = a.get_common_prefix_first_n(ctx, b, n_keep, exact);
    common_prefix ctx_total_discard_prefix = a.get_common_prefix_first_n(ctx, b, n_discard + n_keep, exact);
    // only if there is enough common token
    int32_t discard_offset = ctx_total_discard_prefix.first - (n_discard + n_keep);
    int32_t keep_offset = ctx_keep_prefix.first - n_keep;
    n_kept = ctx_keep_prefix.second - keep_offset;
    n_discarded = ctx_total_discard_prefix.second - ctx_keep_prefix.second - discard_offset;
    if (n_kept < 0) {
        n_kept = n_keep;
    }
    if (n_discarded < 0) {
        n_discarded = n_discard;
    }
}

void server_context::context_shift_prompt(llama_context* ctx, server_slot& slot, bool exact) {
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
            n_discard, n_kept, n_discard_cache, exact);
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

void server_context::update_slots() {
    if (system_need_update) {
        system_prompt_update();
    }

    // release slots
    for (auto& slot : slots) {
        if (slot.command == SLOT_COMMAND_RELEASE) {
            slot.state = SLOT_STATE_IDLE;
            slot.command = SLOT_COMMAND_NONE;
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

        for (auto& slot : slots) {
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
        task.type = SERVER_TASK_TYPE_NEXT_RESPONSE;
        task.id_target = -1;

        queue_tasks.post(std::move(task));
    }

    // apply context-shift if needed
    // TODO: simplify and improve
    for (server_slot& slot : slots) {
        if (slot.ga_n == 1) {
            if (slot.is_processing() && (int)system_tokens.size() + slot.n_past >= slot.n_ctx - 1) {
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

                const int n_left = (int)system_tokens.size() + slot.n_past - n_keep;
                const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);
                int32_t n_kept;
                int32_t n_discard_cache;
                if (n_discard > 0) {
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
    }

    // start populating the batch for this iteration
    llama_batch_clear(batch);

    auto accept_special_token = [&](server_slot& slot, llama_token token) {
        return params.special || slot.sparams.preserved_tokens.find(token) != slot.sparams.preserved_tokens.end();
    };

    // frist, add sampled tokens from any ongoing sequences
    for (auto& slot : slots) {
        if (slot.state == SLOT_STATE_IDLE) {
            continue;
        }

        // generate draft tokens in speculative decoding mode
        // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
        //       perform the speculative drafting for all sequences at the same time in a single batch
        int n_draft_max = slot.get_n_draft_max();
        if (n_draft_max > 0) {
            if (mctx) {
                // we should never reach this, as speculative is automatically disabled if mmproj is loaded
                GGML_ABORT("not supported by multimodal");
            }

            struct llama_speculative_params params_spec;
            params_spec.n_draft = n_draft_max;
            params_spec.n_reuse = llama_n_ctx(slot.ctx_dft) - slot.params.speculative.n_max;
            params_spec.p_min = slot.params.speculative.p_min;
            const llama_tokens& cached_text_tokens = slot.cache_tokens.get_text_tokens();
            llama_tokens draft = llama_speculative_gen_draft(slot.spec, params_spec, cached_text_tokens, slot.sampled);

            // add the sampled token to the batch
            slot.i_batch_dft.push_back(batch.n_tokens);
            llama_batch_add(batch, slot.sampled, slot.cache_tokens.pos_next(), { slot.id }, true);
            slot.cache_tokens.push_back(slot.sampled);

            if (slot.params.speculative.n_min > (int)draft.size()) {
                SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int)draft.size(), slot.params.speculative.n_min);
                // fallback to normal decoding
                slot.i_batch = slot.i_batch_dft[0];
                slot.drafted.clear();
                slot.i_batch_dft.clear();
            }
            else {
                // keep track of total number of drafted tokens tested
                slot.n_draft_total += draft.size();

                // add all drafted tokens to the batch
                for (size_t i = 0; i < draft.size(); i++) {
                    slot.i_batch_dft.push_back(batch.n_tokens);
                    llama_batch_add(batch, draft[i], slot.cache_tokens.pos_next(), { slot.id }, true);
                    slot.cache_tokens.push_back(draft[i]);
                }
                slot.drafted = std::move(draft);
            }
        }
        else {
            // no speculative decoding
            slot.i_batch = batch.n_tokens;

            llama_batch_add(batch, slot.sampled, slot.cache_tokens.pos_next(), { slot.id }, true);

            slot.cache_tokens.push_back(slot.sampled);

            SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                (int)slot.n_ctx, (int)slot.cache_tokens.size(), (int)slot.truncated);
        }
        slot.n_past = slot.cache_tokens.n_tokens();
    }

    // process in chunks of params.n_batch
    int32_t n_batch = llama_n_batch(ctx);
    int32_t n_ubatch = llama_n_ubatch(ctx);

    // track if this is an embedding or non-embedding batch
    // if we've added sampled tokens above, we are in non-embedding mode
    // -1: none, 0: non-embedding, 1: embedding
    int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

    // next, batch any pending prompts without exceeding n_batch
    if (params.cont_batching || batch.n_tokens == 0) {
        for (auto& slot : slots) {
            // this slot still has a prompt to be processed
            if (slot.state == SLOT_STATE_IDLE && slot.command == SLOT_COMMAND_LOAD_PROMPT) {
                auto& prompt_tokens = slot.prompt_tokens;

                // we haven't tokenized the prompt yet - do it now:
                if (prompt_tokens.empty() || slot.n_prompt_tokens == 0) {
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
                    }
                    else {
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
                    }
                    else {
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
                                {"n_left",          slot.n_ctx - slot.params.n_keep},
                                {"n_prompt_tokens", slot.n_prompt_tokens},
                                {"prompt_tokens",   prompt_tokens.detokenize(ctx, true)},
                                });

                            GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);

#ifndef NDEBUG
                            // debug
                            common_prefix prefix = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, false);
                            int32_t back = 1;
                            if (slot.cache_tokens.size() && slot.cache_tokens.size() > prefix.first + 20
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
                            slot.ga_i = 0;
                        }
                        else {
                            GGML_ASSERT(slot.ga_n == 1);

                            // reuse any previously computed tokens that are common with the new prompt
                            common_prefix prefix = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, true); // string level match
                            common_prefix prefix_nonexact = slot.cache_tokens.get_common_prefix(ctx, prompt_tokens, false);
                            auto n_past0 = slot.cache_tokens.get_common_prefix_exact(prompt_tokens); // token level match
                            LLAMA_LOG_INFO("======== Cache: cache_size = %d, n_past0 =  %d, n_past1 =  %d, n_past_prompt1 = %d,  n_past2 =  %d, n_past_prompt2 =  %d\n", (int32_t)slot.cache_tokens.size(), (int32_t)n_past0, (int32_t)prefix.first, (int32_t)prefix.second, (int32_t)prefix_nonexact.first, (int32_t)prefix_nonexact.second);
                            int32_t size_threshold = 20;
                            if (prefix.first + size_threshold < prefix_nonexact.first) {
                                LLAMA_LOG_WARN("Common part contains missing or extra space and new line\n");
                                prefix = prefix_nonexact;
                            }
                            slot.n_past = prefix.first;
                            slot.n_past_prompt = prefix.second;
                            if (slot.n_past != slot.n_past_prompt) {
                                LLAMA_LOG_INFO("Mistokenization found and handled successfully.\n");
                            }
                            if ((slot.n_past + size_threshold < slot.cache_tokens.size()))
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
                }
                else if (batch_type != slot_type) {
                    continue;
                }

                // keep only the common part
                // remove the non-common part from the cache
                slot.cache_tokens.keep_first(slot.n_past);
                int p0 = (int)system_tokens.size() + slot.n_past;
                p0 = system_tokens.size() + slot.cache_tokens.pos_next();
                if (!llama_kv_cache_seq_rm(ctx, slot.id, p0, -1)) {
                    // could not partially delete (likely using a non-Transformer model)
                    llama_kv_cache_seq_rm(ctx, slot.id, -1, -1);

                    p0 = (int)system_tokens.size();
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
                    llama_pos p1 = slot.cache_tokens.pos_next() + slot.n_past_prompt - slot.n_past; // add offset to prompt
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
                            const int bd = (ga_w / ga_n) * (ga_n - 1);
                            slot_npast -= bd;
                            ga_i += ga_w / ga_n;
                        }
                    }

                    int p0 = system_tokens.size() + slot.cache_tokens.pos_next();
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
                    {"progress", (float)slot.n_prompt_tokens_processed / slot.n_prompt_tokens},
                    });

                // entire prompt has been processed - start decoding new tokens
                if (slot.n_past_prompt == slot.n_prompt_tokens) {
                    slot.state = SLOT_STATE_PROCESSING;
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
                    slot.i_batch = batch.n_tokens - 1;

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

        for (auto& slot : slots) {
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
            batch.token + i,
            nullptr,
            batch.pos + i,
            batch.n_seq_id + i,
            batch.seq_id + i,
            batch.logits + i,
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
                for (auto& slot : slots) {
                    slot.state = SLOT_STATE_PROCESSING;
                    slot.command = SLOT_COMMAND_NONE;
                    slot.release();
                    LLAMA_LOG_INFO("n_past = %d\n", (int)slot.cache_tokens.size());
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

        // technically, measuring the time here excludes the sampling time for the last batch
        // but on the other hand, we don't want to do too many system calls to measure the time, so it's ok
        const int64_t t_current = ggml_time_us();

        for (auto& slot : slots) {
            if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
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
            if (slot.i_batch_dft.size() > 0) {
                continue; // sample using speculative decoding
            }
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

            //slot.t_token_generation = (t_current - slot.t_start_generation) / 1e3;
            slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

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

        // speculative decoding - main model sample and accept
        for (auto& slot : slots) {
            if (slot.state != SLOT_STATE_PROCESSING || slot.i_batch_dft.empty()) {
                continue;
            }

            size_t n_draft = slot.drafted.size();

            // the accepted tokens from the speculation
            const auto ids = llama_sampling_sample_and_accept_n(slot.ctx_sampling, ctx, slot.i_batch_dft, slot.drafted);
            slot.i_batch_dft.clear();
            slot.drafted.clear();

            slot.n_past += ids.size();
            slot.n_decoded += ids.size();

            slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += ids.size() - 1;

            // rollback to the state before sampling the draft tokens
            slot.cache_tokens.keep_first(slot.cache_tokens.n_tokens() - n_draft);
            // slot.n_past -= n_draft;
            // add accepted tokens to the prompt
            slot.cache_tokens.insert({ ids.begin(), ids.end() - 1 });
            slot.sampled = ids.back(); // last accepted token
            slot.n_past = slot.cache_tokens.n_tokens();
            llama_kv_cache_seq_rm(ctx, slot.id, slot.n_past, -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok = ids[i];
                result.text_to_send = llama_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob = 1.0f; // set later

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
            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int)ids.size() - 1, (int)slot.drafted.size(), slot.n_past);
            LOG_VERBOSE("speculative decoding result", {
                {"id_slot", slot.id},
                {"accepted", (int)ids.size() - 1},
                {"total", (int)slot.drafted.size()},
                {"new_n_past", slot.n_past}
                });
        }
    }

    LOG_VERBOSE("run slots completed", {});
}

json server_context::model_meta() const {
    return json{
        {"vocab_type",  llama_vocab_type(model)},
        {"n_vocab",     llama_n_vocab(model)},
        {"n_ctx_train", llama_n_ctx_train(model)},
        {"n_embd",      llama_n_embd(model)},
        {"n_params",    llama_model_n_params(model)},
        {"size",        llama_model_size(model)},
    };
}
