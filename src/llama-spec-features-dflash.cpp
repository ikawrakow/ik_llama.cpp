#include "llama-spec-features.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>

#include "llama-model.h"
#include "llama-context.h"

void llama_reset_dflash_kv_cache_state(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dflash.kv.cache_write_pos = 0;
    ctx->dflash.kv.cache_n_filled = 0;
    ctx->dflash.kv.cache_update_rows = 0;
    ctx->dflash.kv.cache_view_write_pos = 0;
    ctx->dflash.kv.cache_view_n_filled = 0;
    ctx->dflash.kv.cache_applied_window_version = 0;
    ctx->dflash.kv.cache_valid = false;
    ctx->dflash.kv.cache_view_valid = false;
    std::fill(ctx->dflash.kv.cache_pos.begin(), ctx->dflash.kv.cache_pos.end(), 0);
    std::fill(ctx->dflash.kv.cache_slot_valid.begin(), ctx->dflash.kv.cache_slot_valid.end(), 0);

    for (ggml_backend_buffer_t buf : ctx->dflash.kv.cache_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_clear(buf, 0);
        }
    }
}

llama_dflash_kv_cache_transition llama_plan_dflash_kv_cache_transition_for_ctx(
        const struct llama_context * ctx,
        const llama_dflash_window_update & window_update,
        int32_t n_rows) {
    if (ctx == nullptr) {
        llama_dflash_kv_cache_transition plan;
        plan.rebuild_cache = true;
        plan.append_rows = std::clamp(window_update.append_rows, 0, n_rows);
        plan.next_n_filled = n_rows;
        return plan;
    }

    const int32_t cross_ctx = ctx->dflash.visible_cross_ctx > 0
            ? ctx->dflash.visible_cross_ctx
            : std::max<int32_t>(1, (int32_t) ctx->cparams.n_ctx - (int32_t) ctx->model.hparams.dflash_block_size);

    return llama_plan_dflash_kv_cache_transition(
            cross_ctx,
            ctx->dflash.kv.cache_n_filled,
            ctx->dflash.kv.cache_write_pos,
            ctx->dflash.kv.cache_valid,
            ctx->dflash.kv.cache_applied_window_version,
            window_update.version,
            window_update.keep_rows,
            window_update.append_rows,
            window_update.replace,
            n_rows);
}

void llama_set_dflash_visible_cross_ctx(
        struct llama_context * ctx,
        int32_t cross_ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dflash.visible_cross_ctx = std::max<int32_t>(0, cross_ctx);
}

int32_t llama_get_dflash_visible_cross_ctx(
        const struct llama_context * ctx) {
    return ctx != nullptr ? ctx->dflash.visible_cross_ctx : 0;
}

int32_t llama_model_dflash_block_size(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_block_size : 0;
}

int32_t llama_model_dflash_mask_token_id(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_mask_token_id : -1;
}

int32_t llama_model_dflash_n_target_layers(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_n_target_layers : 0;
}

int32_t llama_model_dflash_n_target_features(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_n_target_features : 0;
}

int32_t llama_model_dflash_target_layer_ids(
        const struct llama_model * model,
        int32_t * layer_ids,
        int32_t capacity) {
    if (model == nullptr || layer_ids == nullptr || capacity <= 0) {
        return 0;
    }

    const int32_t n_layers = std::min<int32_t>((int32_t) model->hparams.dflash_n_target_layers, capacity);
    for (int32_t i = 0; i < n_layers; ++i) {
        layer_ids[i] = (int32_t) model->hparams.dflash_target_layer_ids[i];
    }

    return n_layers;
}

int32_t llama_model_dflash_target_mask_token_id(const struct llama_model * model) {
    if (model == nullptr) {
        return (int32_t) LLAMA_TOKEN_NULL;
    }

    return (int32_t) model->vocab.token_mask();
}

static const ggml_tensor * llama_dflash_output_tensor(
        const struct llama_model * model) {
    if (model == nullptr) {
        return nullptr;
    }

    if (model->output_mtp != nullptr) {
        return model->output_mtp;
    }

    if (model->output != nullptr) {
        return model->output;
    }

    return model->tok_embd;
}

int32_t llama_model_dflash_io_mode(
        const struct llama_model * draft_model,
        const struct llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr || draft_model->arch != LLM_ARCH_DFLASH_DRAFT) {
        return LLAMA_DFLASH_IO_MODE_INVALID;
    }

    const ggml_tensor * draft_output = llama_dflash_output_tensor(draft_model);
    const ggml_tensor * target_output = llama_dflash_output_tensor(target_model);
    if (draft_model->tok_embd == nullptr || draft_output == nullptr || target_model->tok_embd == nullptr || target_output == nullptr) {
        return LLAMA_DFLASH_IO_MODE_INVALID;
    }

    const bool shared_tok = draft_model->tok_embd == target_model->tok_embd;
    const bool shared_output = draft_output == target_output;
    if (shared_tok && shared_output) {
        return LLAMA_DFLASH_IO_MODE_SHARED;
    }

    if (!shared_tok && !shared_output) {
        return LLAMA_DFLASH_IO_MODE_SELF_CONTAINED;
    }

    return LLAMA_DFLASH_IO_MODE_MIXED;
}

bool llama_model_dflash_io_tensors_match(
        const struct llama_model * draft_model,
        int32_t n_embd,
        int32_t n_vocab) {
    const ggml_tensor * output = llama_dflash_output_tensor(draft_model);
    if (draft_model == nullptr || draft_model->tok_embd == nullptr || output == nullptr || n_embd <= 0 || n_vocab <= 0) {
        return false;
    }

    return (int32_t) draft_model->tok_embd->ne[0] == n_embd &&
           (int32_t) draft_model->tok_embd->ne[1] == n_vocab &&
           (int32_t) output->ne[0] == n_embd &&
           (int32_t) output->ne[1] == n_vocab;
}

bool llama_model_share_dflash_io_tensors(
        struct llama_model * draft_model,
        const struct llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr) {
        return false;
    }

    if (draft_model->arch != LLM_ARCH_DFLASH_DRAFT) {
        return true;
    }

    if (draft_model->tok_embd == nullptr) {
        draft_model->tok_embd = target_model->tok_embd;
    }

    if (draft_model->output == nullptr) {
        draft_model->output = target_model->output ? target_model->output : target_model->tok_embd;
        if (draft_model->output == nullptr) {
            draft_model->output = draft_model->tok_embd;
        }
    }

    const bool uses_shared_tok = draft_model->tok_embd == target_model->tok_embd;
    const bool uses_shared_output = draft_model->output == target_model->output ||
            draft_model->output == target_model->tok_embd;

    if (draft_model->output_mtp == nullptr) {
        if (target_model->output_mtp != nullptr && uses_shared_tok && uses_shared_output) {
            draft_model->output_mtp = target_model->output_mtp;
        } else if (draft_model->output != nullptr) {
            draft_model->output_mtp = draft_model->output;
        } else {
            draft_model->output_mtp = draft_model->tok_embd;
        }
    }

    const struct ggml_tensor * output = llama_dflash_output_tensor(draft_model);
    return draft_model->tok_embd != nullptr && output != nullptr;
}

static bool llama_set_dflash_target_features_impl(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        bool copy_data,
        const llama_dflash_window_update * window_update) {
    const bool have_full_features = target_features != nullptr && n_floats > 0;
    const bool have_append_features = window_update != nullptr &&
            window_update->append_features != nullptr &&
            window_update->append_floats > 0 &&
            window_update->append_rows > 0;

    if (ctx == nullptr || n_rows <= 0 || (!have_full_features && !have_append_features)) {
        return false;
    }

    if (have_full_features && copy_data) {
        ctx->dflash.target.features_owned.assign(target_features, target_features + n_floats);
        ctx->dflash.target.features = ctx->dflash.target.features_owned.data();
    } else if (have_full_features) {
        ctx->dflash.target.features_owned.clear();
        ctx->dflash.target.features = target_features;
    } else {
        ctx->dflash.target.features_owned.clear();
        ctx->dflash.target.features = nullptr;
    }
    ctx->dflash.target.features_n_floats = have_full_features ? n_floats : 0;
    ctx->dflash.target.features_n_rows = n_rows;
    if (have_append_features && copy_data) {
        ctx->dflash.target.append_features_owned.assign(
                window_update->append_features,
                window_update->append_features + window_update->append_floats);
        ctx->dflash.target.append_features = ctx->dflash.target.append_features_owned.data();
    } else if (have_append_features) {
        ctx->dflash.target.append_features_owned.clear();
        ctx->dflash.target.append_features = window_update->append_features;
    } else {
        ctx->dflash.target.append_features_owned.clear();
        ctx->dflash.target.append_features = nullptr;
    }
    ctx->dflash.target.append_features_n_floats = have_append_features ? window_update->append_floats : 0;
    ctx->dflash.target.append_features_n_rows = have_append_features ? window_update->append_rows : 0;
        ctx->dflash.target.version = window_update != nullptr && window_update->version > 0
            ? window_update->version
            : ctx->dflash.target.version + 1;
        ctx->dflash.target.keep_rows = window_update != nullptr
            ? std::max<int32_t>(0, std::min(n_rows, window_update->keep_rows))
            : 0;
        ctx->dflash.target.append_rows = window_update != nullptr
            ? std::max<int32_t>(0, std::min(n_rows, window_update->append_rows))
            : n_rows;
        ctx->dflash.target.replace = window_update != nullptr
            ? window_update->replace
            : true;
        if (ctx->dflash.target.keep_rows + ctx->dflash.target.append_rows > n_rows) {
        ctx->dflash.target.keep_rows = std::max<int32_t>(0, n_rows - ctx->dflash.target.append_rows);
        }

            const int32_t cross_ctx = ctx->dflash.visible_cross_ctx > 0
                ? ctx->dflash.visible_cross_ctx
                : std::max<int32_t>(1, (int32_t) ctx->cparams.n_ctx - (int32_t) ctx->model.hparams.dflash_block_size);
            const llama_dflash_window_update cache_window_update = {
                ctx->dflash.target.version,
                ctx->dflash.target.keep_rows,
                ctx->dflash.target.append_rows,
                ctx->dflash.target.replace,
                ctx->dflash.target.append_features,
                ctx->dflash.target.append_features_n_floats,
            };
            const llama_dflash_kv_cache_transition cache_plan = llama_plan_dflash_kv_cache_transition_for_ctx(ctx, cache_window_update, n_rows);

        if (cache_plan.cache_up_to_date) {
            ctx->dflash.kv.cache_view_n_filled = ctx->dflash.kv.cache_n_filled;
            ctx->dflash.kv.cache_view_write_pos = ctx->dflash.kv.cache_write_pos;
            ctx->dflash.kv.cache_view_valid = ctx->dflash.kv.cache_valid;
        } else if (cross_ctx > 0) {
            ctx->dflash.kv.cache_view_n_filled = cache_plan.next_n_filled;
            ctx->dflash.kv.cache_view_write_pos = cache_plan.next_write_pos;
            ctx->dflash.kv.cache_view_valid = cache_plan.next_n_filled > 0;
        }

    if (target_positions != nullptr) {
        if (copy_data) {
            ctx->dflash.target.positions_owned.assign(target_positions, target_positions + n_rows);
            ctx->dflash.target.positions = ctx->dflash.target.positions_owned.data();
        } else {
            ctx->dflash.target.positions_owned.clear();
            ctx->dflash.target.positions = target_positions;
        }
        ctx->dflash.target.positions_n = (size_t) n_rows;
    } else {
        ctx->dflash.target.positions_owned.clear();
        ctx->dflash.target.positions = nullptr;
        ctx->dflash.target.positions_n = 0;
    }

    return true;
}

bool llama_set_dflash_target_features_copy(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        const llama_dflash_window_update * window_update) {
    return llama_set_dflash_target_features_impl(ctx, target_features, n_floats, n_rows, target_positions, true, window_update);
}

bool llama_set_dflash_target_features_view(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        const llama_dflash_window_update * window_update) {
    return llama_set_dflash_target_features_impl(ctx, target_features, n_floats, n_rows, target_positions, false, window_update);
}

static bool llama_dflash_parse_layer_id(const struct ggml_tensor * tensor, int32_t & layer_id) {
    if (tensor == nullptr) {
        return false;
    }

    static constexpr const char * prefix = "l_out-";
    if (std::strncmp(tensor->name, prefix, std::strlen(prefix)) != 0) {
        return false;
    }

    char * end = nullptr;
    const long raw = std::strtol(tensor->name + std::strlen(prefix), &end, 10);
    if (end == tensor->name + std::strlen(prefix) || *end != '\0') {
        return false;
    }

    layer_id = (int32_t) raw;
    if (layer_id >= 1000) {
        layer_id %= 1000;
    }

    return layer_id >= 0;
}

static int32_t llama_dflash_find_layer_index(const struct llama_context * ctx, int32_t layer_id) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return -1;
    }

    const auto & layer_ids = ctx->dflash.capture->layer_ids;
    const auto it = std::find(layer_ids.begin(), layer_ids.end(), layer_id);
    return it == layer_ids.end() ? -1 : (int32_t) std::distance(layer_ids.begin(), it);
}

static int llama_dflash_capture_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * ctx = static_cast<llama_context *>(user_data);
    if (ctx == nullptr || !ctx->dflash.capture) {
        return false;
    }

    int32_t layer_id = -1;
    if (!llama_dflash_parse_layer_id(tensor, layer_id)) {
        return 0;
    }

    const int32_t layer_idx = llama_dflash_find_layer_index(ctx, layer_id);
    if (layer_idx < 0) {
        return 0;
    }

    //printf("%s -> %d, %d\n", tensor->name, layer_id, layer_idx);

    if (ask) {
        return 2;
    }

    const int32_t row_width = (int32_t) tensor->ne[0];
    const int32_t row_count = row_width > 0 ? (int32_t) (ggml_nelements(tensor) / (int64_t) row_width) : 0;
    if (row_width <= 0 || row_count <= 0) {
        return 0;
    }

    auto & capture = *ctx->dflash.capture;
    if (capture.capture_batch_id == 0) {
        capture.capture_batch_id = 1;
    }
    if (capture.layer_seen_batch_id.size() != capture.layer_ids.size()) {
        capture.layer_seen_batch_id.assign(capture.layer_ids.size(), 0);
    }

    auto & rows = capture.layer_rows[(size_t) layer_idx];
    rows.resize((size_t) row_count * (size_t) row_width);
    auto backend = ggml_backend_sched_get_tensor_backend(ctx->sched, tensor);
    GGML_ASSERT(backend);
    ggml_backend_tensor_get_async(backend, tensor, rows.data(), 0, ggml_nbytes(tensor));
    capture.row_width = row_width;
    capture.row_count = row_count;
    capture.layer_seen_batch_id[(size_t) layer_idx] = capture.capture_batch_id;
    return 2;
}

bool llama_set_dflash_capture_layers(
        struct llama_context * ctx,
        const int32_t * layer_ids,
        int32_t n_layers) {
    if (ctx == nullptr || layer_ids == nullptr || n_layers <= 0) {
        return false;
    }

    auto capture = std::make_unique<llama_context::dflash_runtime::capture_state>();
    capture->layer_ids.assign(layer_ids, layer_ids + n_layers);
    capture->layer_rows.resize((size_t) n_layers);
    capture->layer_seen_batch_id.assign((size_t) n_layers, 0);
    capture->prev_cb_eval = ctx->cparams.cb_eval;
    capture->prev_cb_eval_user_data = ctx->cparams.cb_eval_user_data;
    ctx->dflash.capture = std::move(capture);
    ctx->dflash.feature_view_buffer.clear();

    ctx->cparams.cb_eval = llama_dflash_capture_eval_callback;
    ctx->cparams.cb_eval_user_data = ctx;
    if (ctx->sched != nullptr) {
        ggml_backend_sched_set_eval_callback(ctx->sched, ctx->cparams.cb_eval, ctx->cparams.cb_eval_user_data);
    }

    return true;
}

void llama_clear_dflash_capture(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ggml_backend_sched_eval_callback prev_cb_eval = nullptr;
    void * prev_cb_eval_user_data = nullptr;
    if (ctx->dflash.capture) {
        prev_cb_eval = ctx->dflash.capture->prev_cb_eval;
        prev_cb_eval_user_data = ctx->dflash.capture->prev_cb_eval_user_data;
    }

    ctx->dflash.capture.reset();
    ctx->dflash.feature_view_buffer.clear();

    if (ctx->cparams.cb_eval == llama_dflash_capture_eval_callback && ctx->cparams.cb_eval_user_data == ctx) {
        ctx->cparams.cb_eval = prev_cb_eval;
        ctx->cparams.cb_eval_user_data = prev_cb_eval_user_data;
        if (ctx->sched != nullptr) {
            ggml_backend_sched_set_eval_callback(ctx->sched, prev_cb_eval, prev_cb_eval_user_data);
        }
    }
}

void llama_begin_dflash_capture_batch(struct llama_context * ctx) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return;
    }

    auto & capture = *ctx->dflash.capture;
    capture.capture_batch_id++;
    capture.row_count = 0;
    capture.row_width = 0;
    std::fill(capture.layer_seen_batch_id.begin(), capture.layer_seen_batch_id.end(), 0);
}

void llama_finish_dflash_capture_batch(
        struct llama_context * ctx,
        bool is_prompt_warmup) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return;
    }

    GGML_UNUSED(is_prompt_warmup);
    auto & capture = *ctx->dflash.capture;
    // Reset the batch-local reference shape so the next decode only compares layers within
    // the same batch, not against the previous prompt/verify batch.
    capture.row_count = 0;
    capture.row_width = 0;
}

static bool llama_spec_prepare_dflash_capture(
        struct llama_context * ctx,
        int32_t & row_count,
        int32_t & row_width,
        int32_t & n_layers) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return false;
    }

    llama_synchronize(ctx);

    auto & capture = *ctx->dflash.capture;
    row_count = capture.row_count;
    row_width = capture.row_width;
    n_layers = (int32_t) capture.layer_ids.size();
    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || capture.layer_rows.size() != (size_t) n_layers) {
        return false;
    }

    if (capture.capture_batch_id == 0 || capture.layer_seen_batch_id.size() != (size_t) n_layers) {
        LLAMA_LOG_WARN("%s: DFlash capture batch markers are not initialized (batch_id=%llu layers=%zu expected=%d)\n",
                __func__,
                (unsigned long long) capture.capture_batch_id,
                capture.layer_seen_batch_id.size(),
                n_layers);
        return false;
    }

    for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        if (capture.layer_seen_batch_id[(size_t) layer_idx] != capture.capture_batch_id) {
            LLAMA_LOG_WARN("%s: DFlash capture is stale for layer %d (seen_batch=%llu current_batch=%llu rows=%d width=%d)\n",
                    __func__,
                    capture.layer_ids[(size_t) layer_idx],
                    (unsigned long long) capture.layer_seen_batch_id[(size_t) layer_idx],
                    (unsigned long long) capture.capture_batch_id,
                    row_count,
                    row_width);
            return false;
        }

        const auto & rows = capture.layer_rows[(size_t) layer_idx];
        if (rows.size() != (size_t) row_count * (size_t) row_width) {
            LLAMA_LOG_WARN("%s: DFlash capture rows mismatch for layer %d: got=%zu expected=%zu (rows=%d width=%d)\n",
                    __func__, capture.layer_ids[(size_t) layer_idx], rows.size(),
                    (size_t) row_count * (size_t) row_width, row_count, row_width);
            return false;
        }
    }

    return true;
}

        static bool llama_spec_materialize_dflash_rows_prepared(
            struct llama_context * ctx,
            int32_t row_count,
            int32_t row_width,
            int32_t n_layers,
            const std::vector<int32_t> & row_indices,
            std::vector<float> & rows_out,
            int32_t & combined_width);

static bool llama_spec_materialize_dflash_rows(
        struct llama_context * ctx,
        const std::vector<int32_t> & row_indices,
        std::vector<float> & rows_out,
        int32_t & combined_width) {
    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    return llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, rows_out, combined_width);
}

static bool llama_spec_materialize_dflash_rows_prepared(
        struct llama_context * ctx,
        int32_t row_count,
        int32_t row_width,
        int32_t n_layers,
        const std::vector<int32_t> & row_indices,
        std::vector<float> & rows_out,
        int32_t & combined_width) {
    rows_out.clear();
    combined_width = 0;
    if (ctx == nullptr || row_indices.empty()) {
        return false;
    }

    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || ctx->dflash.capture == nullptr) {
        return false;
    }

    combined_width = row_width * n_layers;
    rows_out.resize((size_t) row_indices.size() * (size_t) combined_width);

    const auto & layer_rows = ctx->dflash.capture->layer_rows;
    for (size_t out_row = 0; out_row < row_indices.size(); ++out_row) {
        int32_t row_index = row_indices[out_row];
        if (row_index < 0) {
            row_index += row_count;
        }
        if (row_index < 0 || row_index >= row_count) {
            rows_out.clear();
            combined_width = 0;
            return false;
        }

        float * dst = rows_out.data() + out_row * (size_t) combined_width;
        for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
            const float * src = layer_rows[(size_t) layer_idx].data() + (size_t) row_index * (size_t) row_width;
            std::memcpy(dst + (size_t) layer_idx * (size_t) row_width, src, (size_t) row_width * sizeof(float));
        }
    }

    return true;
}


bool llama_spec_get_dflash_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view) {
    if (ctx == nullptr || batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    const int32_t batch_row_offset = std::max<int32_t>(0, batch.n_tokens - row_count);
    std::vector<int32_t> row_indices;
    std::vector<int32_t> batch_indices;
    row_indices.reserve((size_t) (batch.n_tokens - batch_row_offset));
    batch_indices.reserve((size_t) (batch.n_tokens - batch_row_offset));
    for (int32_t i = batch_row_offset; i < batch.n_tokens; ++i) {
        row_indices.push_back(i - batch_row_offset);
        batch_indices.push_back(i);
    }

    if (row_indices.empty()) {
        return false;
    }

    view = {};
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    if (!llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, ctx->dflash.feature_view_buffer, view.width)) {
        return false;
    }

    view.rows.reserve(batch_indices.size());
    for (int32_t batch_index : batch_indices) {
        if (batch.n_seq_id[batch_index] <= 0 || batch.seq_id[batch_index] == nullptr) {
            view.rows.clear();
            return false;
        }

        view.rows.push_back({
            /* .seq_id = */ batch.seq_id[batch_index][0],
            /* .pos    = */ batch.pos[batch_index],
            /* .data   = */ ctx->dflash.feature_view_buffer.data() + view.rows.size() * (size_t) view.width,
        });
    }

    return true;
}

bool llama_spec_get_dflash_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view) {
    if (ctx == nullptr || batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    const int32_t batch_row_offset = std::max<int32_t>(0, batch.n_tokens - row_count);
    std::vector<int32_t> row_indices;
    row_indices.reserve((size_t) batch.n_tokens);
    std::vector<int32_t> batch_indices;
    batch_indices.reserve((size_t) batch.n_tokens);
    for (int32_t i = batch_row_offset; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            return false;
        }

        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            if (batch.seq_id[i][j] == seq_id) {
                row_indices.push_back(i - batch_row_offset);
                batch_indices.push_back(i);
                break;
            }
        }
    }

    if (row_indices.empty()) {
        return false;
    }

    view = {};
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    if (!llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, ctx->dflash.feature_view_buffer, view.width)) {
        return false;
    }

    view.rows.reserve(row_indices.size());
    for (size_t i = 0; i < batch_indices.size(); ++i) {
        const int32_t batch_index = batch_indices[i];
        view.rows.push_back({
            /* .seq_id = */ seq_id,
            /* .pos    = */ batch.pos[batch_index],
            /* .data   = */ ctx->dflash.feature_view_buffer.data() + i * (size_t) view.width,
        });
    }

    return true;
}

bool llama_spec_copy_dflash_rows_from_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows) {
    int32_t combined_width = 0;
    if (!llama_spec_materialize_dflash_rows(ctx, output_indices, hidden_rows, combined_width)) {
        hidden_rows.clear();
        return false;
    }

    return hidden_rows.size() == (size_t) output_indices.size() * (size_t) combined_width;
}
