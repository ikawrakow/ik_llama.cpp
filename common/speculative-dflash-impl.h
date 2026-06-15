#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
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

struct common_speculative_state_dflash;
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
        const int32_t target_vocab_size = llama_vocab_n_tokens(vocab_tgt);
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
        LOG_INF("%s: DFlash context ready (n_ctx=%d, block_size=%d, cross_ctx=%d, n_target_features=%d, n_target_layers=%d)\n",
                __func__, llama_n_ctx(ctx_dft), block_size, this->cross_ctx, n_target_features, n_target_layers);
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
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(prompt_tgt);

        result.clear();
        if (!ready || target_window_rows <= 0) {
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
            return;
        }

        llama_kv_cache_clear(ctx_dft);
        batch.n_tokens = 0;
        const int32_t batch_len = n_keep + 1;
        const llama_pos draft_pos_base = last_target_pos >= 0 ? last_target_pos + 1 : (llama_pos) target_window_rows;
        const llama_pos seed_pos = last_target_pos >= 0 ? last_target_pos : draft_pos_base - 1;
        common_batch_add(batch, id_last, seed_pos, { 0 }, false);
        for (int32_t i = 1; i < batch_len; ++i) {
            common_batch_add(batch, mask_token_id, draft_pos_base + (i - 1), { 0 }, i <= n_keep);
        }

        if (llama_decode(ctx_dft, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed for DFlash draft batch\n", __func__);
            batch.n_tokens = 0;
            return;
        }

        result.reserve((size_t) n_keep);
        for (int32_t i = 0; i < n_keep; ++i) {
            llama_token id = llama_get_dflash_draft_token_ith(ctx_dft, i);
            if (id == LLAMA_TOKEN_NULL) {
                id = common_sampler_sample_speculative(nullptr, ctx_dft, i + 1, nullptr);
            }
            result.push_back(id);
        }

        batch.n_tokens = 0;
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
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
        llama_seq_id seq_id) {
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

    for (const auto & row : features.rows) {
        if (row.seq_id != seq_id || row.data == nullptr) {
            continue;
        }

        new_positions.push_back(row.pos);
        new_rows.insert(new_rows.end(), row.data, row.data + row_width);
    }

    if (new_positions.empty()) {
        return false;
    }

    const int32_t n_rows = (int32_t) new_positions.size();
    if (n_rows >= state.cross_ctx) {
        const int32_t keep_from = n_rows - state.cross_ctx;
        state.target_window_pos.assign(new_positions.begin() + keep_from, new_positions.end());
        state.target_window_append_features.assign(
                new_rows.begin() + (ptrdiff_t) keep_from * (ptrdiff_t) row_width,
                new_rows.end());
        dflash_ring_reset_rows(state, state.target_window_append_features.data(), state.cross_ctx);

        state.target_window_rows = state.cross_ctx;
        state.target_window_ring_filled = state.target_window_rows;
        state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
        dflash_record_window_update(state, 0, state.target_window_rows, true);
        return true;
    }

    const int32_t keep_old_rows = std::min<int32_t>(state.target_window_rows, state.cross_ctx - n_rows);
    std::vector<llama_pos> & next_window_pos = state.target_window_pos_stage;
    next_window_pos.resize((size_t) (keep_old_rows + n_rows));

    if (keep_old_rows > 0) {
        std::copy(state.target_window_pos.end() - keep_old_rows, state.target_window_pos.end(), next_window_pos.begin());
    }

    state.target_window_append_features.assign(new_rows.begin(), new_rows.end());
    dflash_ring_append_rows(state, state.target_window_append_features.data(), n_rows);
    std::copy(new_positions.begin(), new_positions.end(), next_window_pos.begin() + keep_old_rows);

    state.target_window_pos.swap(next_window_pos);
    next_window_pos.clear();
    state.target_window_rows = keep_old_rows + n_rows;
    state.target_window_ring_filled = state.target_window_rows;
    state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
    dflash_record_window_update(state, keep_old_rows, n_rows, false);
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
}
