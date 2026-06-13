// DFlash runtime state and draft path.
struct common_speculative_state_dflash : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    llama_batch batch = {};

    int32_t block_size = 0;
    int32_t mask_token_id = -1;
    int32_t n_target_features = 0;
    int32_t cross_ctx = 0;
    bool ready = false;

    std::vector<int32_t> target_layer_ids;
    std::vector<float> target_window;
    std::vector<llama_pos> target_window_pos;
    std::vector<float> target_window_stage;
    std::vector<llama_pos> target_window_pos_stage;
    std::vector<float> target_window_ring;
    std::vector<float> target_window_append_features;
    int32_t target_window_rows = 0;
    int32_t target_window_ring_write_pos = 0;
    int32_t target_window_ring_filled = 0;
    uint64_t target_window_version = 0;
    int32_t target_window_keep_rows = 0;
    int32_t target_window_append_rows = 0;
    bool target_window_replace = false;
    bool target_window_materialized = false;
    llama_pos last_target_pos = -1;
    size_t n_window_updates = 0;
    size_t n_rows_seen = 0;
    size_t n_rows_dropped = 0;
    size_t n_context_shifts = 0;
    size_t n_draft_empty = 0;
    size_t n_set_target_fail = 0;
    size_t n_decode_fail = 0;
    llama_pos last_draft_pos_base = -1;

    uint64_t t_draft_decode_us = 0;
    uint64_t t_draft_sample_us = 0;
    uint64_t t_warmup_collect_us = 0;
    uint64_t t_warmup_append_us = 0;
    uint64_t t_accept_output_copy_us = 0;
    uint64_t t_accept_commit_us = 0;
    uint64_t t_accept_append_us = 0;
    uint64_t t_accept_append_filter_us = 0;
    uint64_t t_accept_append_window_alloc_us = 0;
    uint64_t t_accept_append_replace_us = 0;
    uint64_t t_accept_append_keep_old_us = 0;
    uint64_t t_accept_append_new_rows_us = 0;
    uint64_t t_accept_append_commit_detail_us = 0;
    uint64_t t_accept_append_log_us = 0;
    size_t n_warmup_collect_calls = 0;
    size_t n_warmup_collect_rows = 0;
    size_t n_warmup_append_calls = 0;
    size_t n_warmup_append_rows = 0;
    size_t n_accept_output_copy_calls = 0;
    size_t n_accept_output_copy_rows = 0;
    size_t n_accept_commit_calls = 0;
    size_t n_accept_commit_rows = 0;
    size_t n_accept_append_calls = 0;
    size_t n_accept_append_rows = 0;
    size_t n_accept_append_replace_calls = 0;
    size_t n_accept_append_slide_calls = 0;

    common_speculative_state_dflash(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            int32_t cross_ctx)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , cross_ctx(std::max(1, cross_ctx))
    {
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_dft = llama_get_model(ctx_dft);

        if (!common_speculative_are_dflash_compatible(model_tgt, model_dft)) {
            LOG_ERR("%s: DFlash draft model vocab/tokenizer is incompatible with the target model\n", __func__);
            return;
        }

        block_size = llama_model_dflash_block_size(model_dft);
        mask_token_id = llama_model_dflash_mask_token_id(model_dft);
        n_target_features = llama_model_dflash_n_target_features(model_dft);
        const int32_t n_target_layers = llama_model_dflash_n_target_layers(model_dft);

        if (block_size <= 0 || mask_token_id < 0 || n_target_features <= 0 || n_target_layers <= 0) {
            LOG_ERR("%s: invalid DFlash metadata (block_size=%d, mask_token_id=%d, n_target_features=%d, n_target_layers=%d)\n",
                    __func__, block_size, mask_token_id, n_target_features, n_target_layers);
            return;
        }

        target_layer_ids.resize((size_t) n_target_layers);
        if (llama_model_dflash_target_layer_ids(model_dft, target_layer_ids.data(), n_target_layers) != n_target_layers) {
            LOG_ERR("%s: failed to read DFlash target layer ids\n", __func__);
            target_layer_ids.clear();
            return;
        }

        const auto * vocab_tgt = llama_model_get_vocab(model_tgt);
        const auto * vocab_dft = llama_model_get_vocab(model_dft);
        const int32_t target_vocab_size = llama_vocab_n_tokens(vocab_tgt);
        const int32_t draft_vocab_size = llama_vocab_n_tokens(vocab_dft);
        const int32_t target_hidden_size = llama_model_n_embd(model_tgt);
        const int32_t draft_hidden_size = llama_model_n_embd(model_dft);
        const int32_t target_mask_token_id = llama_model_dflash_target_mask_token_id(model_tgt);
        const int32_t expected_n_target_features = target_hidden_size > 0 ? target_hidden_size * n_target_layers : 0;

        if (target_mask_token_id != (int32_t) LLAMA_TOKEN_NULL && mask_token_id != target_mask_token_id) {
            LOG_ERR("%s: DFlash mask token mismatch (draft=%d target=%d)\n",
                    __func__, mask_token_id, target_mask_token_id);
            return;
        }

        if (target_hidden_size <= 0 || draft_hidden_size <= 0) {
            LOG_ERR("%s: invalid DFlash hidden sizes (draft=%d target=%d)\n",
                    __func__, draft_hidden_size, target_hidden_size);
            return;
        }

        if (expected_n_target_features <= 0 || n_target_features != expected_n_target_features) {
            LOG_ERR("%s: DFlash target feature width mismatch (metadata=%d expected=%d target_hidden=%d target_layers=%d)\n",
                    __func__, n_target_features, expected_n_target_features, target_hidden_size, n_target_layers);
            return;
        }

        std::vector<int32_t> sorted_target_layer_ids = target_layer_ids;
        std::sort(sorted_target_layer_ids.begin(), sorted_target_layer_ids.end());
        if (std::adjacent_find(sorted_target_layer_ids.begin(), sorted_target_layer_ids.end()) != sorted_target_layer_ids.end()) {
            LOG_ERR("%s: duplicate DFlash target layer ids survived into runtime validation\n", __func__);
            target_layer_ids.clear();
            return;
        }

        const int32_t n_target_model_layers = llama_n_layer(model_tgt);
        for (int32_t layer_id : target_layer_ids) {
            if (layer_id < 0 || layer_id >= n_target_model_layers) {
                LOG_ERR("%s: invalid DFlash target layer id %d for target model with %d layers\n",
                        __func__, layer_id, n_target_model_layers);
                target_layer_ids.clear();
                return;
            }
        }

        const int32_t io_mode = llama_model_dflash_io_mode(model_dft, model_tgt);
        if (io_mode == LLAMA_DFLASH_IO_MODE_INVALID) {
            LOG_ERR("%s: DFlash draft is missing required IO tensors after target sharing\n", __func__);
            return;
        }

        if (io_mode == LLAMA_DFLASH_IO_MODE_MIXED) {
            LOG_ERR("%s: DFlash IO contract must be fully shared or fully self-contained, but resolved to mixed mode\n", __func__);
            return;
        }

        if (io_mode == LLAMA_DFLASH_IO_MODE_SELF_CONTAINED && !llama_model_dflash_io_tensors_match(model_dft, target_hidden_size, target_vocab_size)) {
            LOG_ERR("%s: DFlash self-contained IO tensors do not match the target hidden/vocab contract (target_hidden=%d target_vocab=%d)\n",
                    __func__,
                    target_hidden_size,
                    target_vocab_size);
            return;
        }

        if (!llama_set_dflash_capture_layers(ctx_tgt, target_layer_ids.data(), (int32_t) target_layer_ids.size())) {
            LOG_ERR("%s: failed to configure DFlash target capture callback\n", __func__);
            return;
        }

        batch = llama_batch_init(std::max(1, block_size), 0, 1);
        target_window.reserve((size_t) this->cross_ctx * (size_t) n_target_features);
        target_window_stage.reserve((size_t) this->cross_ctx * (size_t) n_target_features);
        target_window_ring.resize((size_t) this->cross_ctx * (size_t) n_target_features);
        target_window_append_features.reserve((size_t) this->cross_ctx * (size_t) n_target_features);
        target_window_pos.reserve((size_t) this->cross_ctx);
        target_window_pos_stage.reserve((size_t) this->cross_ctx);
        ready = true;

        llama_set_dflash_visible_cross_ctx(ctx_dft, this->cross_ctx);
        llama_dflash_profile_reset(ctx_tgt);
        llama_dflash_profile_reset(ctx_dft);

        std::ostringstream layers_oss;
        for (size_t i = 0; i < target_layer_ids.size(); ++i) {
            if (i > 0) {
                layers_oss << ",";
            }
            layers_oss << target_layer_ids[i];
        }

        const char * io_mode_name = io_mode == LLAMA_DFLASH_IO_MODE_SHARED ? "shared" : "self-contained";
        LOG_INF("%s: DFlash context ready (n_ctx=%d, block_size=%d, cross_ctx=%d, n_target_features=%d, target_layer_ids=[%s])\n",
            __func__, llama_n_ctx(ctx_dft), block_size, this->cross_ctx, n_target_features, layers_oss.str().c_str());
        LOG_INF("%s: DFlash artifact io=%s draft_vocab=%d target_vocab=%d draft_hidden=%d target_hidden=%d mask_token_id=%d target_mask_token_id=%d\n",
            __func__, io_mode_name, draft_vocab_size, target_vocab_size, draft_hidden_size, target_hidden_size, mask_token_id, target_mask_token_id);
    }

    ~common_speculative_state_dflash() override {
        llama_clear_dflash_capture(ctx_tgt);
        if (ctx_dft) {
            llama_free(ctx_dft);
        }
        if (batch.token != nullptr) {
            llama_batch_free(batch);
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        llama_kv_cache_clear(ctx_dft);
        llama_reset_dflash_kv_cache_state(ctx_dft);
        n_window_updates = 0;
        n_rows_seen = 0;
        n_rows_dropped = 0;
        n_context_shifts = 0;
        n_draft_empty = 0;
        n_set_target_fail = 0;
        n_decode_fail = 0;
        last_draft_pos_base = -1;
        t_draft_decode_us = 0;
        t_draft_sample_us = 0;
        t_warmup_collect_us = 0;
        t_warmup_append_us = 0;
        t_accept_output_copy_us = 0;
        t_accept_commit_us = 0;
        t_accept_append_us = 0;
        t_accept_append_filter_us = 0;
        t_accept_append_window_alloc_us = 0;
        t_accept_append_replace_us = 0;
        t_accept_append_keep_old_us = 0;
        t_accept_append_new_rows_us = 0;
        t_accept_append_commit_detail_us = 0;
        t_accept_append_log_us = 0;
        n_warmup_collect_calls = 0;
        n_warmup_collect_rows = 0;
        n_warmup_append_calls = 0;
        n_warmup_append_rows = 0;
        n_accept_output_copy_calls = 0;
        n_accept_output_copy_rows = 0;
        n_accept_commit_calls = 0;
        n_accept_commit_rows = 0;
        n_accept_append_calls = 0;
        n_accept_append_rows = 0;
        n_accept_append_replace_calls = 0;
        n_accept_append_slide_calls = 0;
        llama_dflash_profile_reset(ctx_tgt);
        llama_dflash_profile_reset(ctx_dft);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(prompt_tgt);

        result.clear();
        if (!ready || target_window_rows <= 0) {
            n_draft_empty++;
            return;
        }

        const int32_t n_keep = std::min<int32_t>(params.n_max, block_size - 1);
        if (n_keep <= 0) {
            return;
        }

        const float * target_features = nullptr;
        size_t target_feature_floats = 0;
        llama_dflash_window_update window_update = {
            target_window_version,
            target_window_keep_rows,
            target_window_append_rows,
            target_window_replace,
            target_window_append_features.empty() ? nullptr : target_window_append_features.data(),
            target_window_append_features.size(),
        };
        const llama_dflash_kv_cache_transition cache_plan =
                llama_plan_dflash_kv_cache_transition_for_ctx(ctx_dft, window_update, target_window_rows);

        if (cache_plan.rebuild_cache) {
            dflash_materialize_target_window_features(*this);
            target_features = target_window.data();
            target_feature_floats = target_window.size();
            window_update.append_features = target_window.data();
            window_update.append_floats = target_window.size();
            window_update.append_rows = target_window_rows;
        }

        if (!llama_set_dflash_target_features_view(ctx_dft, target_features, target_feature_floats, target_window_rows, target_window_pos.data(), &window_update)) {
            LOG_ERR("%s: failed to set DFlash target features\n", __func__);
            n_set_target_fail++;
            return;
        }

        llama_kv_cache_clear(ctx_dft);
        batch.n_tokens = 0;
        const int32_t batch_len = n_keep + 1;
        const llama_pos draft_pos_base = last_target_pos >= 0 ? last_target_pos + 1 : (llama_pos) target_window_rows;
        const llama_pos seed_pos = last_target_pos >= 0 ? last_target_pos : draft_pos_base - 1;
        last_draft_pos_base = draft_pos_base;
        common_batch_add(batch, id_last, seed_pos, { 0 }, false);
        for (int32_t i = 1; i < batch_len; ++i) {
            common_batch_add(batch, mask_token_id, draft_pos_base + (i - 1), { 0 }, i <= n_keep);
        }

        const int64_t t_decode_us = ggml_time_us();
        if (llama_decode(ctx_dft, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed for DFlash draft batch\n", __func__);
            n_decode_fail++;
            batch.n_tokens = 0;
            return;
        }
        t_draft_decode_us += (uint64_t) (ggml_time_us() - t_decode_us);

        result.reserve((size_t) n_keep);
        const int64_t t_sample_us = ggml_time_us();
        for (int32_t i = 0; i < n_keep; ++i) {
            // Use argmax in GPU when available
            llama_token id = llama_get_dflash_draft_token_ith(ctx_dft, i);
            if (id == LLAMA_TOKEN_NULL) {
                id = common_sampler_sample_speculative(nullptr, ctx_dft, i + 1, nullptr);
            }
            result.push_back(id);
        }
        t_draft_sample_us += (uint64_t) (ggml_time_us() - t_sample_us);

        batch.n_tokens = 0;
        dflash_contract_log_draft(*this, n_keep, result.size());
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};

static void dflash_contract_log_append(
        const common_speculative_state_dflash & state,
        llama_seq_id seq_id,
        const std::vector<llama_pos> & new_positions) {
    if (!dflash_contract_log_enabled()) {
        return;
    }

    static std::atomic<uint64_t> counter = 0;
    const uint64_t ordinal = counter.fetch_add(1, std::memory_order_relaxed);
    if (ordinal >= 8) {
        return;
    }

    const dflash_contract_pos_summary incoming = dflash_contract_summarize_positions(new_positions);
    const dflash_contract_pos_summary window = dflash_contract_summarize_positions(state.target_window_pos);

    LOG_INF("dflash contract append[%llu]: seq=%d incoming_rows=%zu incoming_pos=%s pos=[%d..%d] gaps=%d nonmono=%d window_rows=%d window_pos=%s pos=[%d..%d] gaps=%d nonmono=%d last_target_pos=%d\n",
            (unsigned long long) (ordinal + 1),
            (int) seq_id,
            new_positions.size(),
            dflash_contract_format_values(new_positions).c_str(),
            (int) incoming.first,
            (int) incoming.last,
            incoming.gap_count,
            incoming.nonmono_count,
            state.target_window_rows,
            dflash_contract_format_values(state.target_window_pos).c_str(),
            (int) window.first,
            (int) window.last,
            window.gap_count,
            window.nonmono_count,
            (int) state.last_target_pos);
}

static void dflash_contract_log_draft(
        const common_speculative_state_dflash & state,
        int32_t n_keep,
        size_t result_size) {
    if (!dflash_contract_log_enabled()) {
        return;
    }

    static std::atomic<uint64_t> counter = 0;
    const uint64_t ordinal = counter.fetch_add(1, std::memory_order_relaxed);
    if (ordinal >= 8) {
        return;
    }

    const dflash_contract_pos_summary window = dflash_contract_summarize_positions(state.target_window_pos);
    llama_dflash_profile_stats graph_stats = {};
    llama_dflash_profile_get_stats(state.ctx_dft, &graph_stats);
    const int draft_delta = (state.last_target_pos >= 0 && state.last_draft_pos_base >= 0)
            ? (int) (state.last_draft_pos_base - state.last_target_pos)
            : -1;
        const llama_pos seed_pos = state.last_target_pos;
        const llama_pos mask_first_pos = state.last_draft_pos_base;
        const llama_pos mask_last_pos = state.last_draft_pos_base >= 0
            ? state.last_draft_pos_base + n_keep - 1
            : -1;

        LOG_INF("dflash contract draft[%llu]: window_rows=%d window_pos=%s pos=[%d..%d] gaps=%d nonmono=%d last_target_pos=%d seed_pos=%d mask_pos=[%d..%d] sample_rows=[1..%d] output_rows=[1..%d] draft_pos_base=%d delta=%d n_keep=%d result=%zu set_target(missing/nonmono)=%llu/%llu graph(fallback/nonmono)=%llu/%llu graph_pos=[%d..%d]\n",
            (unsigned long long) (ordinal + 1),
            state.target_window_rows,
            dflash_contract_format_values(state.target_window_pos).c_str(),
            (int) window.first,
            (int) window.last,
            window.gap_count,
            window.nonmono_count,
            (int) state.last_target_pos,
            (int) seed_pos,
            (int) mask_first_pos,
            (int) mask_last_pos,
            n_keep,
            n_keep,
            (int) state.last_draft_pos_base,
            draft_delta,
            n_keep,
            result_size,
            (unsigned long long) graph_stats.set_target_missing_positions,
            (unsigned long long) graph_stats.set_target_non_monotonic_positions,
            (unsigned long long) graph_stats.graph_pos_fallbacks,
            (unsigned long long) graph_stats.graph_pos_non_monotonic,
            (int) graph_stats.last_pos_first,
            (int) graph_stats.last_pos_last);
}

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;
        {
            struct common_params_sampling params;
            params.top_k = 10;
            params.samplers_sequence = {
                llama_sampler_type::TOP_K,
                llama_sampler_type::DIST, // needed to get probabilities
            };
            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);



            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_kv_cache_clear(ctx_dft);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_kv_cache_seq_rm (ctx_dft, 0, 0, reuse_i);
                llama_kv_cache_seq_add(ctx_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_kv_cache_seq_rm (ctx_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, nullptr, id, true);

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                if (i == 0) {
                    result.push_back(id);
                }
                break;
            }

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }


            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;
        n_low = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_suffix : public common_speculative_state {
    common_suffix_tree tree;
    common_suffix_tree corpus_tree;
    bool has_corpus = false;
    size_t cache_size   = 0;

    // Acceptance feedback
    size_t n_draft_last  = 0;
    bool   had_accept    = false;
    int    n_low         = 0;
    float  base_p_min    = 0.1f;
    float  eff_p_min     = 0.1f;

    common_speculative_state_suffix(
            enum common_speculative_type type,
            int max_depth,
            const std::string & corpus_path,
            const llama_model * model)
        : common_speculative_state(type)
        , tree(max_depth)
        , corpus_tree(max_depth)
    {
        if (!corpus_path.empty()) {
            std::function<std::vector<llama_token>(const std::string &)> tokenize_fn;
            if (model) {
                tokenize_fn = [model](const std::string & text) -> std::vector<llama_token> {
                    return common_tokenize(model, text, false, true);
                };
            }
            has_corpus = corpus_tree.load_corpus(corpus_path, tokenize_fn);
        }
    }

    void begin(const llama_tokens & prompt) override {
        cache_size   = 0;
        n_draft_last = 0;
        had_accept   = false;
        n_low        = 0;
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        base_p_min = params.p_min;
        if (n_draft_last > 0 && !had_accept) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        }
        had_accept = false;

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last);

            tree.extend(tokens_new.data(), (int)tokens_new.size());
            cache_size = prompt_tgt.size() + 1;
        }

        const int ctx_len = std::min((int)(prompt_tgt.size() + 1), tree.max_depth());
        llama_tokens context;
        context.reserve(ctx_len);
        const int ctx_start = (int)prompt_tgt.size() + 1 - ctx_len;
        for (int j = ctx_start; j < (int)prompt_tgt.size(); ++j) {
            context.push_back(prompt_tgt[j]);
        }
        context.push_back(id_last);
        const int min_match_len = std::max(1, params.suffix_min_match_len);

        result = tree.speculate(
            context.data(), (int)context.size(),
            params.n_max,
            eff_p_min,
            1,
            min_match_len);

        if (has_corpus) {
            auto corpus_result = corpus_tree.speculate(
                context.data(), (int)context.size(),
                params.n_max,
                eff_p_min,
                1,
                min_match_len);
            if (corpus_result.size() > result.size()) {
                result = std::move(corpus_result);
            }
        }

        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (n_draft_last == 0) {
            return;
        }
        had_accept = true;
        const double f_acc = (double)n_accepted / (double)n_draft_last;
        if (f_acc < 0.5) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        } else {
            n_low = 0;
            if (eff_p_min > base_p_min) {
                eff_p_min = std::max(eff_p_min - 0.05f, base_p_min);
            }
        }
    }
};

struct common_speculative {
    std::vector<common_speculative_config> configs; // resolved stage config for each implementation
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_checkpoint checkpoint;
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
    std::unique_ptr<spec_tuner> tuner;
    int last_n_drafted = 0;
    int64_t t_step_start_us = 0;
};

static bool common_speculative_stage_chain_matches(
        const std::vector<common_speculative_stage_params> & stages,
        const std::vector<common_speculative_config> & configs) {
    if (stages.size() != configs.size()) {
        return false;
    }

    for (size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].type != configs[i].type) {
            return false;
        }
    }

    return true;
}

static common_params_speculative common_speculative_get_runtime_params(
        const common_speculative_config & config,
        const common_params_speculative & params,
        const common_speculative_stage_params & stage) {
    common_params_speculative result = config.params;

    result.type = config.type;
    result.n_max = stage.has_n_max_override() ? stage.n_max : params.n_max;
    result.n_min = stage.has_n_min_override() ? stage.n_min : params.n_min;
    result.p_min = stage.has_p_min_override() ? stage.p_min : params.p_min;

    if (config.type == COMMON_SPECULATIVE_TYPE_SUFFIX) {
        result.suffix_min_match_len = stage.has_suffix_min_match_len_override()
            ? stage.suffix_min_match_len
            : params.suffix_min_match_len;
    }

    result.n_max = std::max(result.n_max, 0);
    result.n_min = std::max(0, std::min(result.n_min, result.n_max));
    result.stages.clear();

    return result;
}

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    std::string normalized = name;
    std::replace(normalized.begin(), normalized.end(), '-', '_');

    const auto it = common_speculative_type_from_name_map.find(normalized);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    bool res = true;

    llama_kv_cache_clear(ctx_tgt);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size(), 0, 0));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_kv_cache_seq_rm(ctx_tgt, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_kv_cache_clear(ctx_tgt);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    std::string chain_error;
    if (!common_speculative_validate_chain(params, &chain_error)) {
        LOG_ERR("%s: invalid speculative stage chain: %s\n", __func__, chain_error.c_str());
        return nullptr;
    }

    const auto stages = params.get_resolved_stages();
    if (params.model_dft && llama_model_is_gemma4_mtp_assistant(params.model_dft)) {
        const bool has_draft_stage = std::any_of(stages.begin(), stages.end(), [](const common_speculative_stage_params & stage) {
            return stage.type == COMMON_SPECULATIVE_TYPE_DRAFT;
        });

        if (has_draft_stage) {
            LOG_ERR("%s: Gemma4 assistant models only support MTP stages; omit -md for self-spec-only runs or use -mtp/--spec-stage mtp for assistant-backed MTP\n", __func__);
            return nullptr;
        }
    }

    const bool has_dflash_stage = std::any_of(stages.begin(), stages.end(), [](const common_speculative_stage_params & stage) {
        return stage.type == COMMON_SPECULATIVE_TYPE_DFLASH;
    });

    const bool needs_draft_ctx = std::any_of(stages.begin(), stages.end(), [&params](const common_speculative_stage_params & stage) {
        return stage.type == COMMON_SPECULATIVE_TYPE_DRAFT ||
               stage.type == COMMON_SPECULATIVE_TYPE_DFLASH ||
               (stage.type == COMMON_SPECULATIVE_TYPE_MTP && params.model_dft != nullptr);
    });

    llama_context * ctx_dft = nullptr;
    if (needs_draft_ctx) {
        if (!params.model_dft) {
            LOG_ERR("%s: draft speculative stage requires a loaded draft model\n", __func__);
            return nullptr;
        }

        llama_context_params cparams_dft = params.cparams_dft;

        if (has_dflash_stage) {
            if (!llama_model_share_dflash_io_tensors(params.model_dft, llama_get_model(ctx_tgt))) {
                LOG_ERR("%s: failed to share target IO tensors with DFlash draft model\n", __func__);
                return nullptr;
            }

            int32_t max_cross_ctx = 0;
            for (const auto & stage : stages) {
                if (stage.type != COMMON_SPECULATIVE_TYPE_DFLASH) {
                    continue;
                }

                max_cross_ctx = std::max(max_cross_ctx, params.with_stage_overrides(stage).dflash_cross_ctx);
            }

            const int32_t block_size = llama_model_dflash_block_size(params.model_dft);
            if (block_size <= 0) {
                LOG_ERR("%s: invalid DFlash draft block size\n", __func__);
                return nullptr;
            }

            const int64_t required_n_ctx = (int64_t) max_cross_ctx + (int64_t) block_size;
            if (required_n_ctx > std::numeric_limits<int32_t>::max()) {
                LOG_ERR("%s: invalid DFlash draft context size cross_ctx=%d block_size=%d required_n_ctx=%lld\n",
                        __func__, max_cross_ctx, block_size, (long long) required_n_ctx);
                return nullptr;
            }

            cparams_dft.n_ctx = (uint32_t) required_n_ctx;
        }

        ctx_dft = llama_init_from_model(params.model_dft, cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the resolved stage chain.
    std::vector<common_speculative_config> configs = {};
    configs.reserve(stages.size());

    for (const auto & stage : stages) {
        common_params_speculative stage_params = params.with_stage_overrides(stage);

        if (stage.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD && !stage_params.ngram_mod) {
            stage_params.ngram_mod = std::make_shared<common_ngram_mod>(stage_params.ngram_size_n, 4*1024*1024);

            LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                    stage_params.ngram_size_n, stage_params.ngram_mod->size(),
                    (float)(stage_params.ngram_mod->size_bytes())/1024/1024);

            if (stage_params.ngram_size_n < 16) {
                LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, stage_params.ngram_size_n);
            }
        }

        configs.push_back(common_speculative_config(stage, stage_params));
    }

    if (!configs.empty() && llama_model_has_recurrent(llama_get_model(ctx_tgt))) {
        const int ckpt_tokens = std::max(1, params.get_max_stage_n_max() + 1);
        const int actual_mode = llama_spec_ckpt_init(ctx_tgt, params.recurrent_ckpt_mode, ckpt_tokens);
        if (actual_mode == LLAMA_SPEC_CKPT_NONE) {
            LOG_ERR("%s: failed to prepare recurrent checkpoint mode '%s' during speculative init (max_tokens=%d)\n",
                    __func__,
                    params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_PER_STEP ? "per-step" :
                    params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_GPU_FALLBACK ? "gpu-fallback" :
                    params.recurrent_ckpt_mode == LLAMA_SPEC_CKPT_CPU ? "cpu" : "auto",
                    ckpt_tokens);
            if (ctx_dft != nullptr) {
                llama_free(ctx_dft);
            }
            return nullptr;
        }
        llama_spec_ckpt_discard(ctx_tgt);
        params.recurrent_ckpt_mode = actual_mode;
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ config.params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                auto state = std::make_unique<common_speculative_state_dflash>(
                    config.type,
                    ctx_tgt,
                    ctx_dft,
                    config.params.dflash_cross_ctx);
                if (!state->ready) {
                    LOG_ERR("%s: failed to initialize DFlash speculative state\n", __func__);
                    return nullptr;
                }
                impls.push_back(std::move(state));
                ctx_dft = nullptr;
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                llama_context * ctx_mtp = ctx_dft;
                if (!ctx_mtp) {
                    const llama_model * model = llama_get_model(ctx_tgt);
                    ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model), config.params.cparams_dft);
                    if (!ctx_mtp) {
                        LOG_ERR("%s: failed to create MTP context\n", __func__);
                        return nullptr;
                    }
                }
                ctx_dft = nullptr;

                const bool use_constant_draft_positions = llama_model_is_gemma4_mtp_assistant(llama_get_model(ctx_mtp));
                impls.push_back(std::make_unique<common_speculative_state_mtp>(
                    config.type, ctx_tgt, ctx_mtp, use_constant_draft_positions));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config.params.lookup_cache_static, config.params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                int depth = config.params.suffix_max_depth > 0 ? config.params.suffix_max_depth : 64;
                const llama_model * model = llama_get_model(ctx_tgt);
                impls.push_back(std::make_unique<common_speculative_state_suffix>(
                    config.type, depth, config.params.suffix_corpus, model));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .configs = */ std::move(configs),
        /* .impls = */ std::move(impls)
    };

    // initialize autotune if requested
    if (params.autotune && params.has_composite_stage_chain()) {
        LOG_WRN("Autotune disabled — explicit speculative stage chains are not supported yet\n");
    } else if (params.autotune && !result->impls.empty()) {
        auto actual_type = result->impls[0]->type;
        if (actual_type != COMMON_SPECULATIVE_TYPE_NONE &&
            actual_type != COMMON_SPECULATIVE_TYPE_EAGLE3) {
            result->tuner = std::make_unique<spec_tuner>();
            result->tuner->init(actual_type, params, llama_get_model(ctx_tgt));
            LOG_DBG("Autotune initialized for %s, tuning %zu parameters\n",
                    common_speculative_type_to_str(actual_type).c_str(),
                    result->tuner->coords.size());
        } else {
            LOG_WRN("Autotune disabled — speculative type %s is not supported for autotuning\n",
                    common_speculative_type_to_str(actual_type).c_str());
        }
    }

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    spec->checkpoint.clear();
    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last,
        llama_pos draft_base_pos,
        llama_seq_id draft_seq_id) {
    llama_tokens result;

    spec->t_step_start_us = ggml_time_us();

    // apply autotune proposal if enabled
    if (spec->tuner && spec->tuner->enabled) {
        spec->tuner->propose(params);
    }

    const auto runtime_stages = params.get_resolved_stages();
    const bool use_runtime_stage_overrides = common_speculative_stage_chain_matches(runtime_stages, spec->configs);

    spec->curr_impl = nullptr; // reset current implementation

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto & impl = spec->impls[i];
        const auto & runtime_stage = use_runtime_stage_overrides ? runtime_stages[i] : spec->configs[i].stage;
        common_params_speculative impl_params = common_speculative_get_runtime_params(spec->configs[i], params, runtime_stage);
        result.clear();

        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(impl_params, prompt_tgt, id_last, draft_base_pos, draft_seq_id, result);
            impl->n_call_draft++;
        }

        if (result.empty()) {
            continue;
        }

        if (common_speculative_type_is_self_spec(impl->type) && impl_params.n_min > 0 && (int)result.size() < impl_params.n_min) {
            LOG_DBG("%s: impl %s drafted %zu tokens, below fallback threshold %d - trying next implementation\n",
                    __func__, common_speculative_type_to_str(impl->type).c_str(), result.size(), impl_params.n_min);
            result.clear();
            continue;
        }
        LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                impl.get()->n_call_draft, result.size());

        spec->curr_impl = impl.get();
        impl->n_gen_drafts++;
        impl->n_gen_tokens += result.size();

        break; // We have a draft, so break out of the loop and return it.
    }

    // store draft count for tuner feedback
    if (spec->tuner && spec->tuner->enabled) {
        spec->last_n_drafted = (int)result.size();
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (spec->tuner && spec->tuner->enabled && spec->t_step_start_us > 0) {
        int64_t step_time_us = ggml_time_us() - spec->t_step_start_us;
        double step_tps = (step_time_us > 100)
            ? (n_accepted + 1.0) * 1e6 / (double)step_time_us
            : 0.0;
        spec->tuner->accept_feedback(n_accepted, spec->last_n_drafted, step_tps);
        spec->t_step_start_us = 0;
    }

    common_speculative_state * impl = spec->curr_impl;

    if (!impl) {
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }

    if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
        if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
            mtp_invalidate_cached_drafts(*mtp_state);
        }
    }
}

static bool common_speculative_has_type(const common_speculative * spec, common_speculative_type type) {
    if (spec == nullptr) {
        return false;
    }

    return std::any_of(spec->configs.begin(), spec->configs.end(), [type](const common_speculative_config & config) {
        return config.type == type;
    });
}

static int common_speculative_ctx_mtp_n_embd(llama_context * ctx) {
    return ctx ? (int) llama_mtp_state_n_embd(ctx) : 0;
}

static bool common_speculative_batch_token_has_seq_id(
        const llama_batch & batch,
        int token_index,
        llama_seq_id seq_id) {
    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[token_index] <= 0 || batch.seq_id[token_index] == nullptr) {
        return false;
    }

    for (int i = 0; i < batch.n_seq_id[token_index]; ++i) {
        if (batch.seq_id[token_index][i] == seq_id) {
            return true;
        }
    }

    return false;
}

static bool common_speculative_batch_is_exact_single_seq(
        const llama_batch & batch,
        llama_seq_id seq_id) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return false;
        }
    }

    return true;
}

static int common_speculative_copy_seq_batch(
        const llama_batch & batch,
        llama_seq_id seq_id,
        llama_batch & seq_batch) {
    if (batch.token == nullptr || batch.pos == nullptr) {
        return -1;
    }

    if (batch.n_tokens < 1) {
        return 0;
    }

    std::vector<int> token_indices;
    token_indices.reserve(batch.n_tokens);
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (common_speculative_batch_token_has_seq_id(batch, i, seq_id)) {
            token_indices.push_back(i);
        }
    }

    if (token_indices.empty()) {
        return 0;
    }

    seq_batch = llama_batch_init((int) token_indices.size(), 0, 1);
    for (const int i : token_indices) {
        common_batch_add(seq_batch, batch.token[i], batch.pos[i], { seq_id }, batch.logits != nullptr && batch.logits[i]);
    }

    return (int) token_indices.size();
}

static bool common_speculative_feature_view_copy_batch_rows(
        const common_speculative_feature_view & view,
        const llama_batch & batch,
        llama_seq_id seq_id,
        std::vector<float> * hidden_rows) {
    if (hidden_rows == nullptr || view.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || view.width <= 0 || batch.n_tokens <= 0 || batch.pos == nullptr) {
        return false;
    }

    std::unordered_map<llama_pos, const float *> rows_by_pos;
    rows_by_pos.reserve(view.rows.size());
    for (const auto & row : view.rows) {
        if (row.seq_id == seq_id && row.data != nullptr) {
            rows_by_pos[row.pos] = row.data;
        }
    }

    hidden_rows->clear();
    hidden_rows->reserve((size_t) batch.n_tokens * view.width);
    for (int i = 0; i < batch.n_tokens; ++i) {
        auto it = rows_by_pos.find(batch.pos[i]);
        if (it == rows_by_pos.end()) {
            hidden_rows->clear();
            return false;
        }

        hidden_rows->insert(hidden_rows->end(), it->second, it->second + view.width);
    }

    return hidden_rows->size() == (size_t) batch.n_tokens * view.width;
}
