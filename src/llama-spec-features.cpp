#include "llama-spec-features.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>

#include "llama-model.h"
#include "llama-context.h"

static bool llama_dflash_positions_strictly_increasing(
        const llama_pos * positions,
        int32_t n_rows,
        llama_pos & first_pos,
        llama_pos & last_pos) {
    first_pos = -1;
    last_pos = -1;

    if (positions == nullptr || n_rows <= 0) {
        return false;
    }

    first_pos = positions[0];
    last_pos = positions[n_rows - 1];

    for (int32_t i = 1; i < n_rows; ++i) {
        if (positions[i] <= positions[i - 1]) {
            return false;
        }
    }

    return true;
}

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }

    const auto & hparams = ctx->model.hparams;
    if (ctx->cparams.mtp && ctx->model.arch == LLM_ARCH_GEMMA4_MTP && hparams.mtp_backbone_n_embd > 0) {
        return hparams.mtp_backbone_n_embd;
    }

    return hparams.n_embd;
}

void llama_dflash_profile_reset(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dflash_profile = {};
}

void llama_reset_dflash_kv_cache_state(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dflash_kv_cache_write_pos = 0;
    ctx->dflash_kv_cache_n_filled = 0;
    ctx->dflash_kv_cache_update_rows = 0;
    ctx->dflash_kv_cache_view_write_pos = 0;
    ctx->dflash_kv_cache_view_n_filled = 0;
    ctx->dflash_kv_cache_applied_window_version = 0;
    ctx->dflash_kv_cache_valid = false;
    ctx->dflash_kv_cache_view_valid = false;
    ctx->dflash_kv_workspace_write_pos = 0;
    ctx->dflash_kv_workspace_n_filled = 0;
    ctx->dflash_kv_workspace_applied_window_version = 0;
    ctx->dflash_kv_workspace_valid = false;
    ctx->dflash_kv_workspace_sync_pending = false;

    for (ggml_backend_buffer_t buf : ctx->dflash_cache_bufs) {
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

    const int32_t cross_ctx = ctx->dflash_visible_cross_ctx > 0
            ? ctx->dflash_visible_cross_ctx
            : std::max<int32_t>(1, (int32_t) ctx->cparams.n_ctx - (int32_t) ctx->model.hparams.dflash_block_size);

    return llama_plan_dflash_kv_cache_transition(
            cross_ctx,
            ctx->dflash_kv_cache_n_filled,
            ctx->dflash_kv_cache_write_pos,
            ctx->dflash_kv_cache_valid,
            ctx->dflash_kv_cache_applied_window_version,
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

    ctx->dflash_visible_cross_ctx = std::max<int32_t>(0, cross_ctx);
}

int32_t llama_get_dflash_visible_cross_ctx(
        const struct llama_context * ctx) {
    return ctx != nullptr ? ctx->dflash_visible_cross_ctx : 0;
}

bool llama_dflash_profile_get_stats(
        const struct llama_context * ctx,
        llama_dflash_profile_stats * stats) {
    if (ctx == nullptr || stats == nullptr) {
        return false;
    }

    *stats = ctx->dflash_profile;
    return true;
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

const struct ggml_tensor * llama_model_dflash_output_tensor(
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

static const char * llama_dflash_io_mode_name(int32_t io_mode) {
    switch (io_mode) {
        case LLAMA_DFLASH_IO_MODE_SHARED:
            return "shared";
        case LLAMA_DFLASH_IO_MODE_SELF_CONTAINED:
            return "self-contained";
        case LLAMA_DFLASH_IO_MODE_MIXED:
            return "mixed";
        default:
            return "invalid";
    }
}

static const char * llama_dflash_output_head_kind(
        const struct llama_model * draft_model,
        const struct llama_model * target_model) {
    const struct ggml_tensor * output = llama_model_dflash_output_tensor(draft_model);
    if (output == nullptr) {
        return "missing";
    }

    if (output == draft_model->tok_embd) {
        return draft_model->tok_embd == (target_model ? target_model->tok_embd : nullptr)
                ? "shared_token_embedding"
                : "token_embedding";
    }

    if (draft_model->output_mtp != nullptr && output == draft_model->output_mtp) {
        if (target_model != nullptr && target_model->output_mtp != nullptr && output == target_model->output_mtp) {
            return "output_mtp";
        }

        if (std::strcmp(output->name, "output_extra.weight") == 0) {
            return "output_extra";
        }

        return "output_mtp";
    }

    return "output";
}

int32_t llama_model_dflash_io_mode(
        const struct llama_model * draft_model,
        const struct llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr || draft_model->arch != LLM_ARCH_DFLASH_DRAFT) {
        return LLAMA_DFLASH_IO_MODE_INVALID;
    }

    const ggml_tensor * draft_output = llama_model_dflash_output_tensor(draft_model);
    const ggml_tensor * target_output = llama_model_dflash_output_tensor(target_model);
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
    const ggml_tensor * output = llama_model_dflash_output_tensor(draft_model);
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

    if (draft_model->output_mtp == nullptr && target_model->output_mtp != nullptr && uses_shared_tok && uses_shared_output) {
        draft_model->output_mtp = target_model->output_mtp;
    }

    const struct ggml_tensor * output = llama_model_dflash_output_tensor(draft_model);
    if (draft_model->tok_embd != nullptr && output != nullptr) {
        LLAMA_LOG_INFO("%s: DFlash IO mode=%s output_head=%s tensor=%s type=%s\n",
                __func__,
                llama_dflash_io_mode_name(llama_model_dflash_io_mode(draft_model, target_model)),
                llama_dflash_output_head_kind(draft_model, target_model),
                output->name[0] != '\0' ? output->name : "(unnamed)",
                ggml_type_name(output->type));
    }

    return draft_model->tok_embd != nullptr && output != nullptr;
}

bool llama_set_draft_input_hidden_state_copy(
        struct llama_context * ctx,
        const float * hidden_state,
        size_t n_floats) {
    if (ctx == nullptr || hidden_state == nullptr || n_floats == 0) {
        return false;
    }

    ctx->draft_input_hidden_state_owned.assign(hidden_state, hidden_state + n_floats);
    ctx->draft_input_hidden_state = ctx->draft_input_hidden_state_owned.data();
    ctx->draft_input_hidden_state_n_floats = n_floats;
    return true;
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

    auto & profile = ctx->dflash_profile;
    const int64_t t_start_us = ggml_time_us();
    const int32_t row_width = have_full_features
            ? (n_rows > 0 ? (int32_t) (n_floats / (size_t) n_rows) : 0)
            : (window_update->append_rows > 0 ? (int32_t) (window_update->append_floats / (size_t) window_update->append_rows) : 0);
    llama_pos first_pos = -1;
    llama_pos last_pos = -1;

    if (have_full_features && copy_data) {
        ctx->dflash_target_features_owned.assign(target_features, target_features + n_floats);
        ctx->dflash_target_features = ctx->dflash_target_features_owned.data();
    } else if (have_full_features) {
        ctx->dflash_target_features_owned.clear();
        ctx->dflash_target_features = target_features;
    } else {
        ctx->dflash_target_features_owned.clear();
        ctx->dflash_target_features = nullptr;
    }
    ctx->dflash_target_features_n_floats = have_full_features ? n_floats : 0;
    ctx->dflash_target_features_n_rows = n_rows;
    if (have_append_features && copy_data) {
        ctx->dflash_target_append_features_owned.assign(
                window_update->append_features,
                window_update->append_features + window_update->append_floats);
        ctx->dflash_target_append_features = ctx->dflash_target_append_features_owned.data();
    } else if (have_append_features) {
        ctx->dflash_target_append_features_owned.clear();
        ctx->dflash_target_append_features = window_update->append_features;
    } else {
        ctx->dflash_target_append_features_owned.clear();
        ctx->dflash_target_append_features = nullptr;
    }
    ctx->dflash_target_append_features_n_floats = have_append_features ? window_update->append_floats : 0;
    ctx->dflash_target_append_features_n_rows = have_append_features ? window_update->append_rows : 0;
        ctx->dflash_target_window_version = window_update != nullptr && window_update->version > 0
            ? window_update->version
            : ctx->dflash_target_window_version + 1;
        ctx->dflash_target_window_keep_rows = window_update != nullptr
            ? std::max<int32_t>(0, std::min(n_rows, window_update->keep_rows))
            : 0;
        ctx->dflash_target_window_append_rows = window_update != nullptr
            ? std::max<int32_t>(0, std::min(n_rows, window_update->append_rows))
            : n_rows;
        ctx->dflash_target_window_replace = window_update != nullptr
            ? window_update->replace
            : true;
        if (ctx->dflash_target_window_keep_rows + ctx->dflash_target_window_append_rows > n_rows) {
        ctx->dflash_target_window_keep_rows = std::max<int32_t>(0, n_rows - ctx->dflash_target_window_append_rows);
        }

            const int32_t cross_ctx = ctx->dflash_visible_cross_ctx > 0
                ? ctx->dflash_visible_cross_ctx
                : std::max<int32_t>(1, (int32_t) ctx->cparams.n_ctx - (int32_t) ctx->model.hparams.dflash_block_size);
            const llama_dflash_window_update cache_window_update = {
                ctx->dflash_target_window_version,
                ctx->dflash_target_window_keep_rows,
                ctx->dflash_target_window_append_rows,
                ctx->dflash_target_window_replace,
                ctx->dflash_target_append_features,
                ctx->dflash_target_append_features_n_floats,
            };
            const llama_dflash_kv_cache_transition cache_plan = llama_plan_dflash_kv_cache_transition_for_ctx(ctx, cache_window_update, n_rows);

        if (cache_plan.cache_up_to_date) {
            ctx->dflash_kv_cache_view_n_filled = ctx->dflash_kv_cache_n_filled;
            ctx->dflash_kv_cache_view_write_pos = ctx->dflash_kv_cache_write_pos;
            ctx->dflash_kv_cache_view_valid = ctx->dflash_kv_cache_valid;
        } else if (cross_ctx > 0) {
            ctx->dflash_kv_cache_view_n_filled = cache_plan.next_n_filled;
            ctx->dflash_kv_cache_view_write_pos = cache_plan.next_write_pos;
            ctx->dflash_kv_cache_view_valid = cache_plan.next_n_filled > 0;
        }

    if (target_positions != nullptr) {
        if (copy_data) {
            ctx->dflash_target_positions_owned.assign(target_positions, target_positions + n_rows);
            ctx->dflash_target_positions = ctx->dflash_target_positions_owned.data();
        } else {
            ctx->dflash_target_positions_owned.clear();
            ctx->dflash_target_positions = target_positions;
        }
        ctx->dflash_target_positions_n = (size_t) n_rows;
    } else {
        ctx->dflash_target_positions_owned.clear();
        ctx->dflash_target_positions = nullptr;
        ctx->dflash_target_positions_n = 0;
    }

    profile.set_target_copy_calls++;
    profile.set_target_copy_us += (uint64_t) (ggml_time_us() - t_start_us);
    profile.set_target_rows += (uint64_t) n_rows;
        profile.set_target_copy_bytes +=
            (have_full_features ? n_floats : 0) * sizeof(float) +
            (have_append_features ? window_update->append_floats : 0) * sizeof(float) +
            (target_positions ? (size_t) n_rows * sizeof(llama_pos) : 0);
    profile.last_n_rows = n_rows;
    profile.last_width = row_width;

    if (target_positions == nullptr) {
        profile.set_target_missing_positions++;
        profile.last_pos_first = -1;
        profile.last_pos_last = -1;
    } else {
        if (!llama_dflash_positions_strictly_increasing(target_positions, n_rows, first_pos, last_pos)) {
            profile.set_target_non_monotonic_positions++;
        }
        profile.last_pos_first = first_pos;
        profile.last_pos_last = last_pos;
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

static void llama_record_dflash_capture_phase(
        struct llama_context * ctx,
        bool is_prompt_warmup,
        int32_t row_count,
        int32_t row_width) {
    if (ctx == nullptr || row_count <= 0 || row_width <= 0) {
        return;
    }

    auto & profile = ctx->dflash_profile;
    if (is_prompt_warmup) {
        profile.capture_prompt_batches++;
        if (profile.capture_prompt_last_rows > 0 && profile.capture_prompt_last_width > 0 &&
                (profile.capture_prompt_last_rows != row_count || profile.capture_prompt_last_width != row_width)) {
            profile.capture_prompt_shape_changes++;
        }
        profile.capture_prompt_last_rows = row_count;
        profile.capture_prompt_last_width = row_width;
    } else {
        profile.capture_verify_batches++;
        if (profile.capture_verify_last_rows > 0 && profile.capture_verify_last_width > 0 &&
                (profile.capture_verify_last_rows != row_count || profile.capture_verify_last_width != row_width)) {
            profile.capture_verify_shape_changes++;
        }
        profile.capture_verify_last_rows = row_count;
        profile.capture_verify_last_width = row_width;
    }
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
    if (ctx == nullptr || !ctx->dflash_capture) {
        return -1;
    }

    const auto & layer_ids = ctx->dflash_capture->layer_ids;
    const auto it = std::find(layer_ids.begin(), layer_ids.end(), layer_id);
    return it == layer_ids.end() ? -1 : (int32_t) std::distance(layer_ids.begin(), it);
}

static bool llama_dflash_capture_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * ctx = static_cast<llama_context *>(user_data);
    if (ctx == nullptr || !ctx->dflash_capture) {
        return false;
    }

    int32_t layer_id = -1;
    if (!llama_dflash_parse_layer_id(tensor, layer_id)) {
        return false;
    }

    const int32_t layer_idx = llama_dflash_find_layer_index(ctx, layer_id);
    if (layer_idx < 0) {
        return false;
    }

    if (ask) {
        return true;
    }

    const int32_t row_width = (int32_t) tensor->ne[0];
    const int32_t row_count = row_width > 0 ? (int32_t) (ggml_nelements(tensor) / (int64_t) row_width) : 0;
    if (row_width <= 0 || row_count <= 0) {
        return false;
    }

    auto & capture = *ctx->dflash_capture;
    if (capture.capture_batch_id == 0) {
        capture.capture_batch_id = 1;
    }
    if (capture.layer_seen_batch_id.size() != capture.layer_ids.size()) {
        capture.layer_seen_batch_id.assign(capture.layer_ids.size(), 0);
    }

    auto & rows = capture.layer_rows[(size_t) layer_idx];
    rows.resize((size_t) row_count * (size_t) row_width);
    ggml_backend_tensor_get(tensor, rows.data(), 0, ggml_nbytes(tensor));
    capture.row_width = row_width;
    capture.row_count = row_count;
    capture.layer_seen_batch_id[(size_t) layer_idx] = capture.capture_batch_id;
    return true;
}

bool llama_set_dflash_capture_layers(
        struct llama_context * ctx,
        const int32_t * layer_ids,
        int32_t n_layers) {
    if (ctx == nullptr || layer_ids == nullptr || n_layers <= 0) {
        return false;
    }

    auto capture = std::make_unique<llama_context::dflash_capture_state>();
    capture->layer_ids.assign(layer_ids, layer_ids + n_layers);
    capture->layer_rows.resize((size_t) n_layers);
    capture->layer_seen_batch_id.assign((size_t) n_layers, 0);
    capture->prev_cb_eval = ctx->cparams.cb_eval;
    capture->prev_cb_eval_user_data = ctx->cparams.cb_eval_user_data;
    ctx->dflash_capture = std::move(capture);
    ctx->dflash_feature_view_buffer.clear();

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
    if (ctx->dflash_capture) {
        prev_cb_eval = ctx->dflash_capture->prev_cb_eval;
        prev_cb_eval_user_data = ctx->dflash_capture->prev_cb_eval_user_data;
    }

    ctx->dflash_capture.reset();
    ctx->dflash_feature_view_buffer.clear();

    if (ctx->cparams.cb_eval == llama_dflash_capture_eval_callback && ctx->cparams.cb_eval_user_data == ctx) {
        ctx->cparams.cb_eval = prev_cb_eval;
        ctx->cparams.cb_eval_user_data = prev_cb_eval_user_data;
        if (ctx->sched != nullptr) {
            ggml_backend_sched_set_eval_callback(ctx->sched, prev_cb_eval, prev_cb_eval_user_data);
        }
    }
}

void llama_begin_dflash_capture_batch(struct llama_context * ctx) {
    if (ctx == nullptr || !ctx->dflash_capture) {
        return;
    }

    auto & capture = *ctx->dflash_capture;
    capture.capture_batch_id++;
    capture.row_count = 0;
    capture.row_width = 0;
    std::fill(capture.layer_seen_batch_id.begin(), capture.layer_seen_batch_id.end(), 0);
}

void llama_finish_dflash_capture_batch(
        struct llama_context * ctx,
        bool is_prompt_warmup) {
    if (ctx == nullptr || !ctx->dflash_capture) {
        return;
    }

    auto & capture = *ctx->dflash_capture;
    llama_record_dflash_capture_phase(ctx, is_prompt_warmup, capture.row_count, capture.row_width);

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
    if (ctx == nullptr || !ctx->dflash_capture) {
        return false;
    }

    auto & profile = ctx->dflash_profile;
    profile.capture_prepare_calls++;
    const int64_t t_sync_us = ggml_time_us();
    llama_synchronize(ctx);
    profile.capture_prepare_sync_us += (uint64_t) (ggml_time_us() - t_sync_us);

    auto & capture = *ctx->dflash_capture;
    row_count = capture.row_count;
    row_width = capture.row_width;
    n_layers = (int32_t) capture.layer_ids.size();
    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || capture.layer_rows.size() != (size_t) n_layers) {
        profile.capture_prepare_failures++;
        return false;
    }

    if (capture.capture_batch_id == 0 || capture.layer_seen_batch_id.size() != (size_t) n_layers) {
        profile.capture_prepare_failures++;
        profile.capture_layer_batch_mismatch++;
        if (profile.capture_layer_batch_mismatch <= 3) {
            LLAMA_LOG_WARN("%s: DFlash capture batch markers are not initialized (batch_id=%llu layers=%zu expected=%d)\n",
                    __func__,
                    (unsigned long long) capture.capture_batch_id,
                    capture.layer_seen_batch_id.size(),
                    n_layers);
        }
        return false;
    }

    for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        if (capture.layer_seen_batch_id[(size_t) layer_idx] != capture.capture_batch_id) {
            profile.capture_prepare_failures++;
            profile.capture_layer_batch_mismatch++;
            if (profile.capture_layer_batch_mismatch <= 3) {
                LLAMA_LOG_WARN("%s: DFlash capture is stale for layer %d (seen_batch=%llu current_batch=%llu rows=%d width=%d)\n",
                        __func__,
                        capture.layer_ids[(size_t) layer_idx],
                        (unsigned long long) capture.layer_seen_batch_id[(size_t) layer_idx],
                        (unsigned long long) capture.capture_batch_id,
                        row_count,
                        row_width);
            }
            return false;
        }

        const auto & rows = capture.layer_rows[(size_t) layer_idx];
        if (rows.size() != (size_t) row_count * (size_t) row_width) {
            profile.capture_prepare_failures++;
            profile.capture_layer_shape_mismatch++;
            if (profile.capture_layer_shape_mismatch <= 3) {
                LLAMA_LOG_WARN("%s: DFlash capture rows mismatch for layer %d: got=%zu expected=%zu (rows=%d width=%d)\n",
                        __func__, capture.layer_ids[(size_t) layer_idx], rows.size(),
                        (size_t) row_count * (size_t) row_width, row_count, row_width);
            }
            return false;
        }
    }

    return true;
}

static bool llama_dflash_contract_log_enabled() {
    const char * env = std::getenv("IK_DFLASH_CONTRACT_LOG");
    if (env == nullptr || *env == '\0') {
        return false;
    }

    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "off") != 0;
}

template <typename T>
static std::string llama_dflash_contract_format_values(
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

static std::vector<llama_pos> llama_dflash_contract_collect_batch_positions(
        const llama_batch & batch,
        const std::vector<int32_t> & batch_indices) {
    std::vector<llama_pos> positions;
    positions.reserve(batch_indices.size());
    for (int32_t batch_index : batch_indices) {
        positions.push_back(batch.pos[batch_index]);
    }
    return positions;
}

static void llama_dflash_contract_summarize_positions(
        const std::vector<llama_pos> & positions,
        llama_pos & first_pos,
        llama_pos & last_pos,
        int32_t & gap_count,
        int32_t & nonmono_count) {
    first_pos = -1;
    last_pos = -1;
    gap_count = 0;
    nonmono_count = 0;
    if (positions.empty()) {
        return;
    }

    first_pos = positions.front();
    last_pos = positions.back();
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] <= positions[i - 1]) {
            nonmono_count++;
        } else if (positions[i] != positions[i - 1] + 1) {
            gap_count++;
        }
    }
}

static void llama_dflash_contract_log_feature_view(
        const char * kind,
        llama_seq_id seq_id,
        const llama_batch & batch,
        int32_t row_count,
        int32_t row_width,
        int32_t n_layers,
        int32_t batch_row_offset,
        const std::vector<int32_t> & row_indices,
        const std::vector<int32_t> & batch_indices) {
    if (!llama_dflash_contract_log_enabled()) {
        return;
    }

    static std::atomic<uint64_t> counter = 0;
    const uint64_t ordinal = counter.fetch_add(1, std::memory_order_relaxed);
    if (ordinal >= 8) {
        return;
    }

    const std::vector<llama_pos> positions = llama_dflash_contract_collect_batch_positions(batch, batch_indices);
    llama_pos first_pos = -1;
    llama_pos last_pos = -1;
    int32_t gap_count = 0;
    int32_t nonmono_count = 0;
    llama_dflash_contract_summarize_positions(positions, first_pos, last_pos, gap_count, nonmono_count);

    LLAMA_LOG_INFO("%s[%llu]: kind=%s seq=%d batch_tokens=%d capture_rows=%d row_width=%d layers=%d batch_row_offset=%d row_indices=%s batch_indices=%s batch_pos=%s pos=[%d..%d] gaps=%d nonmono=%d\n",
            __func__,
            (unsigned long long) (ordinal + 1),
            kind,
            (int) seq_id,
            batch.n_tokens,
            row_count,
            row_width,
            n_layers,
            batch_row_offset,
            llama_dflash_contract_format_values(row_indices).c_str(),
            llama_dflash_contract_format_values(batch_indices).c_str(),
            llama_dflash_contract_format_values(positions).c_str(),
            (int) first_pos,
            (int) last_pos,
            gap_count,
            nonmono_count);
}

static void llama_dflash_contract_log_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices) {
    if (!llama_dflash_contract_log_enabled()) {
        return;
    }

    static std::atomic<uint64_t> counter = 0;
    const uint64_t ordinal = counter.fetch_add(1, std::memory_order_relaxed);
    if (ordinal >= 8) {
        return;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    const bool have_capture = llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers);

    LLAMA_LOG_INFO("%s[%llu]: output_indices=%s capture_rows=%d row_width=%d layers=%d have_capture=%s\n",
            __func__,
            (unsigned long long) (ordinal + 1),
            llama_dflash_contract_format_values(output_indices).c_str(),
            row_count,
            row_width,
            n_layers,
            have_capture ? "true" : "false");
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
        if (ctx != nullptr) {
            ctx->dflash_profile.capture_materialize_failures++;
        }
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

    auto & profile = ctx->dflash_profile;
    profile.capture_materialize_calls++;
    const int64_t t_start_us = ggml_time_us();

    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || ctx->dflash_capture == nullptr) {
        profile.capture_materialize_failures++;
        return false;
    }

    combined_width = row_width * n_layers;
    rows_out.resize((size_t) row_indices.size() * (size_t) combined_width);

    const auto & layer_rows = ctx->dflash_capture->layer_rows;
    for (size_t out_row = 0; out_row < row_indices.size(); ++out_row) {
        int32_t row_index = row_indices[out_row];
        if (row_index < 0) {
            row_index += row_count;
        }
        if (row_index < 0 || row_index >= row_count) {
            rows_out.clear();
            combined_width = 0;
            profile.capture_materialize_failures++;
            return false;
        }

        float * dst = rows_out.data() + out_row * (size_t) combined_width;
        for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
            const float * src = layer_rows[(size_t) layer_idx].data() + (size_t) row_index * (size_t) row_width;
            std::memcpy(dst + (size_t) layer_idx * (size_t) row_width, src, (size_t) row_width * sizeof(float));
        }
    }

    profile.capture_materialize_us += (uint64_t) (ggml_time_us() - t_start_us);
    profile.capture_materialize_rows += (uint64_t) row_indices.size();
    profile.capture_materialize_bytes += rows_out.size() * sizeof(float);

    return true;
}

static bool llama_spec_prepare_hidden_feature_view(
        struct llama_context   * ctx,
        int32_t                  n_rows,
        llama_spec_feature_view & view) {
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    view.width = 0;
    view.rows.clear();

    if (ctx == nullptr || n_rows < 0) {
        return false;
    }

    llama_synchronize(ctx);

    if (ctx->embd == nullptr) {
        return false;
    }

    view.width = (int32_t) llama_mtp_state_n_embd(ctx);
    if (view.width <= 0 || ctx->n_outputs_embd < n_rows) {
        view.width = 0;
        return false;
    }

    view.rows.reserve(n_rows);
    return true;
}

bool llama_spec_get_hidden_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view) {
    if (batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    if (!llama_spec_prepare_hidden_feature_view(ctx, batch.n_tokens, view)) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            view.rows.clear();
            return false;
        }

        view.rows.push_back({
            /* .seq_id = */ batch.seq_id[i][0],
            /* .pos    = */ batch.pos[i],
            /* .data   = */ ctx->embd + (size_t) i * view.width,
        });
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
    if (!llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, ctx->dflash_feature_view_buffer, view.width)) {
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
            /* .data   = */ ctx->dflash_feature_view_buffer.data() + view.rows.size() * (size_t) view.width,
        });
    }

    llama_dflash_contract_log_feature_view(
            "batch",
            view.rows.empty() ? -1 : view.rows.front().seq_id,
            batch,
            row_count,
            row_width,
            n_layers,
            batch_row_offset,
            row_indices,
            batch_indices);

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
    if (!llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, ctx->dflash_feature_view_buffer, view.width)) {
        return false;
    }

    view.rows.reserve(row_indices.size());
    for (size_t i = 0; i < batch_indices.size(); ++i) {
        const int32_t batch_index = batch_indices[i];
        view.rows.push_back({
            /* .seq_id = */ seq_id,
            /* .pos    = */ batch.pos[batch_index],
            /* .data   = */ ctx->dflash_feature_view_buffer.data() + i * (size_t) view.width,
        });
    }

    llama_dflash_contract_log_feature_view(
            "seq",
            seq_id,
            batch,
            row_count,
            row_width,
            n_layers,
            batch_row_offset,
            row_indices,
            batch_indices);

    return true;
}

bool llama_spec_get_hidden_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view) {
    if (batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    if (!llama_spec_prepare_hidden_feature_view(ctx, batch.n_tokens, view)) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            view.rows.clear();
            return false;
        }

        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            if (batch.seq_id[i][j] != seq_id) {
                continue;
            }

            view.rows.push_back({
                /* .seq_id = */ seq_id,
                /* .pos    = */ batch.pos[i],
                /* .data   = */ ctx->embd + (size_t) i * view.width,
            });
            break;
        }
    }

    return !view.rows.empty();
}

bool llama_spec_get_hidden_feature_view_from_output_index(
        struct llama_context   * ctx,
        int32_t                  output_index,
        llama_seq_id             seq_id,
        llama_pos                pos,
        llama_spec_feature_view & view) {
    if (!llama_spec_prepare_hidden_feature_view(ctx, 1, view)) {
        return false;
    }

    if (output_index < 0) {
        output_index += ctx->n_outputs_embd;
    }
    if (output_index < 0 || output_index >= ctx->n_outputs_embd) {
        view.rows.clear();
        return false;
    }

    view.rows.push_back({
        /* .seq_id = */ seq_id,
        /* .pos    = */ pos,
        /* .data   = */ ctx->embd + (size_t) output_index * view.width,
    });
    return true;
}

bool llama_spec_copy_hidden_rows_from_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows) {
    hidden_rows.clear();
    if (output_indices.empty()) {
        return false;
    }

    llama_spec_feature_view view;
    if (!llama_spec_prepare_hidden_feature_view(ctx, (int32_t) output_indices.size(), view)) {
        return false;
    }

    hidden_rows.reserve((size_t) output_indices.size() * view.width);
    for (int32_t output_index : output_indices) {
        if (output_index < 0) {
            output_index += ctx->n_outputs_embd;
        }
        if (output_index < 0 || output_index >= ctx->n_outputs_embd) {
            hidden_rows.clear();
            return false;
        }

        const float * row = ctx->embd + (size_t) output_index * view.width;
        hidden_rows.insert(hidden_rows.end(), row, row + view.width);
    }

    return hidden_rows.size() == (size_t) output_indices.size() * view.width;
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

    llama_dflash_contract_log_output_indices(ctx, output_indices);

    return hidden_rows.size() == (size_t) output_indices.size() * (size_t) combined_width;
}
