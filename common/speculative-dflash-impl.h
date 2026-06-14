#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

static bool common_speculative_are_dflash_compatible(
        const llama_model * model_tgt,
        const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft)) {
        LOG_DBG("%s: DFlash draft model vocab type must match the target model\n", __func__);
        return false;
    }

    const bool add_bos_tgt = llama_vocab_get_add_bos(vocab_tgt);
    const bool add_bos_dft = llama_vocab_get_add_bos(vocab_dft);
    const bool add_eos_tgt = llama_vocab_get_add_eos(vocab_tgt);
    const bool add_eos_dft = llama_vocab_get_add_eos(vocab_dft);
    const llama_token bos_tgt = llama_vocab_bos(vocab_tgt);
    const llama_token bos_dft = llama_vocab_bos(vocab_dft);
    const llama_token eos_tgt = llama_vocab_eos(vocab_tgt);
    const llama_token eos_dft = llama_vocab_eos(vocab_dft);

    if (add_bos_tgt != add_bos_dft || add_eos_tgt != add_eos_dft ||
        (add_bos_tgt && bos_tgt != bos_dft) ||
        (add_eos_tgt && eos_tgt != eos_dft)) {
        LOG_DBG("%s: DFlash draft special tokens must match the target model (add_bos=%d/%d add_eos=%d/%d bos=%d/%d eos=%d/%d)\n",
                __func__,
                (int) add_bos_tgt,
                (int) add_bos_dft,
                (int) add_eos_tgt,
                (int) add_eos_dft,
                (int) bos_tgt,
                (int) bos_dft,
                (int) eos_tgt,
                (int) eos_dft);
        return false;
    }

    const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
    const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
    const int vocab_diff  = n_vocab_tgt > n_vocab_dft
        ? n_vocab_tgt - n_vocab_dft
        : n_vocab_dft - n_vocab_tgt;

    if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
        LOG_DBG("%s: DFlash draft vocab size differs too much from the target model (%d vs %d)\n",
                __func__, n_vocab_dft, n_vocab_tgt);
        return false;
    }

    for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
        const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
        const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

        if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
            LOG_DBG("%s: DFlash draft token %d differs - target '%s', draft '%s'\n", __func__, i,
                    common_token_to_piece(vocab_tgt, i).c_str(),
                    common_token_to_piece(vocab_dft, i).c_str());
            return false;
        }
    }

    return true;
}

static bool dflash_contract_log_enabled() {
    const char * env = std::getenv("IK_DFLASH_CONTRACT_LOG");
    if (env == nullptr || *env == '\0') {
        return false;
    }

    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "off") != 0;
}

static bool dflash_stats_log_enabled() {
    const char * env = std::getenv("IK_DFLASH_STATS_LOG");
    if (env == nullptr || *env == '\0') {
        return false;
    }

    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "off") != 0;
}

template <typename T>
static std::string dflash_contract_format_values(
        const std::vector<T> & values,
        size_t edge_count = 4) {
    std::ostringstream oss;
    oss << '[';
    if (values.empty()) {
        oss << ']';
        return oss.str();
    }

    const size_t head = std::min(edge_count, values.size());
    for (size_t i = 0; i < head; ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << values[i];
    }

    if (values.size() > edge_count * 2) {
        oss << ",...,";
        for (size_t i = values.size() - edge_count; i < values.size(); ++i) {
            if (i > values.size() - edge_count) {
                oss << ',';
            }
            oss << values[i];
        }
    } else {
        for (size_t i = head; i < values.size(); ++i) {
            oss << ',' << values[i];
        }
    }

    oss << ']';
    return oss.str();
}

struct dflash_contract_pos_summary {
    llama_pos first = -1;
    llama_pos last = -1;
    int32_t gap_count = 0;
    int32_t nonmono_count = 0;
};

static dflash_contract_pos_summary dflash_contract_summarize_positions(
        const std::vector<llama_pos> & positions) {
    dflash_contract_pos_summary summary;
    if (positions.empty()) {
        return summary;
    }

    summary.first = positions.front();
    summary.last = positions.back();
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] <= positions[i - 1]) {
            summary.nonmono_count++;
        } else if (positions[i] != positions[i - 1] + 1) {
            summary.gap_count++;
        }
    }

    return summary;
}

struct common_speculative_state_dflash;

static void dflash_contract_log_append(
        const common_speculative_state_dflash & state,
        llama_seq_id seq_id,
        const std::vector<llama_pos> & new_positions);
static void dflash_contract_log_draft(
        const common_speculative_state_dflash & state,
        int32_t n_keep,
        size_t result_size);
static void dflash_materialize_target_window_features(common_speculative_state_dflash & state);

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

struct dflash_append_breakdown {
    uint64_t filter_us = 0;
    uint64_t window_alloc_us = 0;
    uint64_t replace_us = 0;
    uint64_t keep_old_us = 0;
    uint64_t new_rows_us = 0;
    uint64_t commit_us = 0;
    uint64_t log_us = 0;
    bool replace_call = false;
};

static void dflash_record_window_update(
        common_speculative_state_dflash & state,
        int32_t keep_rows,
        int32_t append_rows,
        bool replace) {
    state.target_window_keep_rows = std::max<int32_t>(0, keep_rows);
    state.target_window_append_rows = std::max<int32_t>(0, append_rows);
    state.target_window_replace = replace;
    state.target_window_version++;
}

static void dflash_ring_reset_rows(
        common_speculative_state_dflash & state,
        const float * rows,
        int32_t n_rows) {
    const size_t row_width = (size_t) state.n_target_features;
    if (n_rows <= 0 || rows == nullptr) {
        state.target_window_ring_write_pos = 0;
        state.target_window_ring_filled = 0;
        return;
    }

    if (state.target_window_ring.size() != (size_t) state.cross_ctx * row_width) {
        state.target_window_ring.resize((size_t) state.cross_ctx * row_width);
    }

    std::memcpy(state.target_window_ring.data(), rows, (size_t) n_rows * row_width * sizeof(float));
    state.target_window_ring_write_pos = n_rows % state.cross_ctx;
    state.target_window_ring_filled = n_rows;
    state.target_window_materialized = false;
}

static void dflash_ring_append_rows(
        common_speculative_state_dflash & state,
        const float * rows,
        int32_t n_rows) {
    const size_t row_width = (size_t) state.n_target_features;
    if (n_rows <= 0 || rows == nullptr) {
        return;
    }

    if (state.target_window_ring.size() != (size_t) state.cross_ctx * row_width) {
        state.target_window_ring.resize((size_t) state.cross_ctx * row_width);
    }

    int32_t write_pos = state.target_window_ring_write_pos;
    int32_t remaining = n_rows;
    const float * src = rows;
    while (remaining > 0) {
        const int32_t chunk_rows = std::min<int32_t>(remaining, state.cross_ctx - write_pos);
        std::memcpy(
                state.target_window_ring.data() + (size_t) write_pos * row_width,
                src,
                (size_t) chunk_rows * row_width * sizeof(float));
        src += (size_t) chunk_rows * row_width;
        remaining -= chunk_rows;
        write_pos = (write_pos + chunk_rows) % state.cross_ctx;
    }

    state.target_window_ring_write_pos = write_pos;
    state.target_window_ring_filled = std::min(state.cross_ctx, state.target_window_ring_filled + n_rows);
    state.target_window_materialized = false;
}

static void dflash_materialize_target_window_features(common_speculative_state_dflash & state) {
    if (state.target_window_materialized || state.target_window_rows <= 0) {
        return;
    }

    const size_t row_width = (size_t) state.n_target_features;
    state.target_window.resize((size_t) state.target_window_rows * row_width);

    const int32_t read_start = (state.target_window_ring_write_pos - state.target_window_rows + state.cross_ctx) % state.cross_ctx;
    const int32_t first_rows = std::min<int32_t>(state.target_window_rows, state.cross_ctx - read_start);
    std::memcpy(
            state.target_window.data(),
            state.target_window_ring.data() + (size_t) read_start * row_width,
            (size_t) first_rows * row_width * sizeof(float));

    const int32_t second_rows = state.target_window_rows - first_rows;
    if (second_rows > 0) {
        std::memcpy(
                state.target_window.data() + (size_t) first_rows * row_width,
                state.target_window_ring.data(),
                (size_t) second_rows * row_width * sizeof(float));
    }

    state.target_window_materialized = true;
}

static bool dflash_append_target_features(
        common_speculative_state_dflash & state,
        const common_speculative_feature_view & features,
        const llama_batch & batch,
        llama_seq_id seq_id,
        dflash_append_breakdown * breakdown = nullptr) {
    GGML_UNUSED(batch);

    if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE ||
            features.width != state.n_target_features ||
            features.rows.empty() ||
            state.cross_ctx <= 0) {
        return false;
    }

    const size_t row_width = (size_t) state.n_target_features;
    std::vector<float> new_rows;
    std::vector<llama_pos> new_positions;
    new_rows.reserve(features.rows.size() * row_width);
    new_positions.reserve(features.rows.size());

    const int64_t t_filter_us = ggml_time_us();
    for (const auto & row : features.rows) {
        if (row.seq_id != seq_id || row.data == nullptr) {
            continue;
        }

        new_positions.push_back(row.pos);
        new_rows.insert(new_rows.end(), row.data, row.data + row_width);
    }
    if (breakdown != nullptr) {
        breakdown->filter_us += (uint64_t) (ggml_time_us() - t_filter_us);
    }

    if (new_positions.empty()) {
        return false;
    }

    const int32_t n_rows = (int32_t) new_positions.size();
    state.n_window_updates++;
    state.n_rows_seen += (size_t) n_rows;
    if (n_rows >= state.cross_ctx) {
        state.n_rows_dropped += (size_t) state.target_window_rows + (size_t) (n_rows - state.cross_ctx);
        const int32_t keep_from = n_rows - state.cross_ctx;
        const int64_t t_replace_us = ggml_time_us();
        state.target_window_pos.assign(new_positions.begin() + keep_from, new_positions.end());
        state.target_window_append_features.assign(
                new_rows.begin() + (ptrdiff_t) keep_from * (ptrdiff_t) row_width,
                new_rows.end());
        dflash_ring_reset_rows(state, state.target_window_append_features.data(), state.cross_ctx);
        if (breakdown != nullptr) {
            breakdown->replace_us += (uint64_t) (ggml_time_us() - t_replace_us);
            breakdown->replace_call = true;
        }

        const int64_t t_commit_us = ggml_time_us();
        state.target_window_rows = state.cross_ctx;
        state.target_window_ring_filled = state.target_window_rows;
        state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
        dflash_record_window_update(state, 0, state.target_window_rows, true);
        if (breakdown != nullptr) {
            breakdown->commit_us += (uint64_t) (ggml_time_us() - t_commit_us);
        }

        const int64_t t_log_us = ggml_time_us();
        dflash_contract_log_append(state, seq_id, new_positions);
        if (breakdown != nullptr) {
            breakdown->log_us += (uint64_t) (ggml_time_us() - t_log_us);
        }
        return true;
    }

    const int32_t keep_old_rows = std::min<int32_t>(state.target_window_rows, state.cross_ctx - n_rows);
    state.n_rows_dropped += (size_t) std::max<int32_t>(0, state.target_window_rows - keep_old_rows);
    const int64_t t_window_alloc_us = ggml_time_us();
    std::vector<llama_pos> & next_window_pos = state.target_window_pos_stage;
    next_window_pos.resize((size_t) (keep_old_rows + n_rows));
    if (breakdown != nullptr) {
        breakdown->window_alloc_us += (uint64_t) (ggml_time_us() - t_window_alloc_us);
    }

    if (keep_old_rows > 0) {
        const int64_t t_keep_old_us = ggml_time_us();
        std::copy(state.target_window_pos.end() - keep_old_rows, state.target_window_pos.end(), next_window_pos.begin());
        if (breakdown != nullptr) {
            breakdown->keep_old_us += (uint64_t) (ggml_time_us() - t_keep_old_us);
        }
    }

    const int64_t t_new_rows_us = ggml_time_us();
    state.target_window_append_features.assign(new_rows.begin(), new_rows.end());
    dflash_ring_append_rows(state, state.target_window_append_features.data(), n_rows);
    std::copy(new_positions.begin(), new_positions.end(), next_window_pos.begin() + keep_old_rows);
    if (breakdown != nullptr) {
        breakdown->new_rows_us += (uint64_t) (ggml_time_us() - t_new_rows_us);
    }

    const int64_t t_commit_us = ggml_time_us();
    state.target_window_pos.swap(next_window_pos);
    next_window_pos.clear();
    state.target_window_rows = keep_old_rows + n_rows;
    state.target_window_ring_filled = state.target_window_rows;
    state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
    dflash_record_window_update(state, keep_old_rows, n_rows, false);
    if (breakdown != nullptr) {
        breakdown->commit_us += (uint64_t) (ggml_time_us() - t_commit_us);
    }

    const int64_t t_log_us = ggml_time_us();
    dflash_contract_log_append(state, seq_id, new_positions);
    if (breakdown != nullptr) {
        breakdown->log_us += (uint64_t) (ggml_time_us() - t_log_us);
    }
    return true;
}

static void dflash_clear_target_features(common_speculative_state_dflash & state) {
    state.target_window.clear();
    state.target_window_pos.clear();
    state.target_window_stage.clear();
    state.target_window_pos_stage.clear();
    state.target_window_append_features.clear();
    state.target_window_rows = 0;
    state.target_window_ring_write_pos = 0;
    state.target_window_ring_filled = 0;
    state.target_window_keep_rows = 0;
    state.target_window_append_rows = 0;
    state.target_window_replace = false;
    state.target_window_materialized = false;
    state.last_target_pos = -1;
    llama_reset_dflash_kv_cache_state(state.ctx_dft);
}

static void dflash_context_shift(
        common_speculative_state_dflash & state,
        llama_pos kv_keep,
        llama_pos kv_discard,
        llama_pos kv_past) {
    if (kv_discard <= 0 || state.target_window_rows <= 0 || state.target_window_pos.empty()) {
        return;
    }

    dflash_materialize_target_window_features(state);

    const size_t row_width = (size_t) state.n_target_features;
    const llama_pos discard_begin = kv_keep;
    const llama_pos discard_end = kv_keep + kv_discard;

    std::vector<float> shifted_rows;
    std::vector<llama_pos> shifted_positions;
    shifted_rows.reserve(state.target_window.size());
    shifted_positions.reserve(state.target_window_pos.size());

    for (int32_t row = 0; row < state.target_window_rows; ++row) {
        llama_pos pos = state.target_window_pos[(size_t) row];
        if (pos >= discard_begin && pos < discard_end) {
            continue;
        }

        if (pos >= discard_end && pos < kv_past) {
            pos -= kv_discard;
        }

        const float * row_src = state.target_window.data() + (size_t) row * row_width;
        shifted_rows.insert(shifted_rows.end(), row_src, row_src + row_width);
        shifted_positions.push_back(pos);
    }

    state.target_window = std::move(shifted_rows);
    state.target_window_pos = std::move(shifted_positions);
    state.target_window_rows = (int32_t) state.target_window_pos.size();
    dflash_ring_reset_rows(state, state.target_window.data(), state.target_window_rows);
    state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
    dflash_record_window_update(state, 0, state.target_window_rows, true);
    llama_reset_dflash_kv_cache_state(state.ctx_dft);
    state.n_context_shifts++;
}
