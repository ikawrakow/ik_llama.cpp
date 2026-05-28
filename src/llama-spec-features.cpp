#include "llama-spec-features.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>

#include "llama-model.h"
#include "llama-context.h"

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

    return draft_model->tok_embd != nullptr && draft_model->output != nullptr;
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

bool llama_set_dflash_target_features_copy(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows) {
    if (ctx == nullptr || target_features == nullptr || n_floats == 0 || n_rows <= 0) {
        return false;
    }

    ctx->dflash_target_features_owned.assign(target_features, target_features + n_floats);
    ctx->dflash_target_features = ctx->dflash_target_features_owned.data();
    ctx->dflash_target_features_n_floats = n_floats;
    ctx->dflash_target_features_n_rows = n_rows;
    return true;
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
    auto & rows = capture.layer_rows[(size_t) layer_idx];
    rows.resize((size_t) row_count * (size_t) row_width);
    ggml_backend_tensor_get(tensor, rows.data(), 0, ggml_nbytes(tensor));
    capture.row_width = row_width;
    capture.row_count = row_count;
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

static bool llama_spec_prepare_dflash_capture(
        struct llama_context * ctx,
        int32_t & row_count,
        int32_t & row_width,
        int32_t & n_layers) {
    if (ctx == nullptr || !ctx->dflash_capture) {
        return false;
    }

    llama_synchronize(ctx);

    auto & capture = *ctx->dflash_capture;
    row_count = capture.row_count;
    row_width = capture.row_width;
    n_layers = (int32_t) capture.layer_ids.size();
    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || capture.layer_rows.size() != (size_t) n_layers) {
        return false;
    }

    for (const auto & rows : capture.layer_rows) {
        if (rows.size() != (size_t) row_count * (size_t) row_width) {
            return false;
        }
    }

    return true;
}

static bool llama_spec_materialize_dflash_rows(
        struct llama_context * ctx,
        const std::vector<int32_t> & row_indices,
        std::vector<float> & rows_out,
        int32_t & combined_width) {
    rows_out.clear();
    combined_width = 0;
    if (ctx == nullptr || row_indices.empty()) {
        return false;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
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

    std::vector<int32_t> row_indices((size_t) batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        row_indices[(size_t) i] = i;
    }

    view = {};
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    if (!llama_spec_materialize_dflash_rows(ctx, row_indices, ctx->dflash_feature_view_buffer, view.width)) {
        return false;
    }

    view.rows.reserve((size_t) batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            view.rows.clear();
            return false;
        }

        view.rows.push_back({
            /* .seq_id = */ batch.seq_id[i][0],
            /* .pos    = */ batch.pos[i],
            /* .data   = */ ctx->dflash_feature_view_buffer.data() + (size_t) i * (size_t) view.width,
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

    std::vector<int32_t> row_indices;
    row_indices.reserve((size_t) batch.n_tokens);
    std::vector<int32_t> batch_indices;
    batch_indices.reserve((size_t) batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            return false;
        }

        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            if (batch.seq_id[i][j] == seq_id) {
                row_indices.push_back(i);
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
    if (!llama_spec_materialize_dflash_rows(ctx, row_indices, ctx->dflash_feature_view_buffer, view.width)) {
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

    return hidden_rows.size() == (size_t) output_indices.size() * (size_t) combined_width;
}
