#include "llama-spec-features.h"

#include <random>

#include "llama-model.h"
#include "llama-context.h"

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }

    const auto & hparams = ctx->model.hparams;
    if (ctx->cparams.mtp && (ctx->model.arch == LLM_ARCH_GEMMA4_MTP || ctx->model.arch == LLM_ARCH_GEMMA4_ASSISTANT) && hparams.mtp_backbone_n_embd > 0) {
        return hparams.mtp_backbone_n_embd;
    }

    return hparams.n_embd;
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
