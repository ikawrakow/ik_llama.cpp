#include "llama-spec-features.h"

#include <random>

#include "llama-model.h"
#include "llama-context.h"

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }

    if (!ctx->cparams.mtp) {
        return ctx->model.hparams.n_embd;
    }

    return llama_model_mtp_feature_width(&ctx->model);
}

uint32_t llama_model_mtp_feature_width(const struct llama_model * model) {
    if (model == nullptr) {
        return 0;
    }

    const auto & hparams = model->hparams;
    if ((model->arch == LLM_ARCH_GEMMA4_MTP || model->arch == LLM_ARCH_GEMMA4_ASSISTANT) &&
        hparams.mtp_backbone_n_embd > 0) {
        return hparams.mtp_backbone_n_embd;
    }
    if ((model->arch == LLM_ARCH_DEEPSEEK4 ||
         model->arch == LLM_ARCH_QWEN4EXP) &&
        hparams.n_embd_out > hparams.n_embd) {
        // the pre-final-mixer wide stream; n_embd_out holds even when the NextN block lives in a companion file
        return hparams.n_embd_out;
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

static bool llama_model_qwen4exp_io_needs_clone(const ggml_tensor * tensor, ggml_backend_buffer_type_t buft) {
    return tensor != nullptr && tensor->buffer != nullptr && buft != nullptr &&
           ggml_backend_buffer_get_type(tensor->buffer) != buft;
}

static ggml_tensor * llama_model_clone_qwen4exp_io_tensor(
        llama_model * model,
        ggml_tensor * source,
        ggml_backend_buffer_type_t buft,
        std::unique_ptr<ggml_tensor> & storage,
        const char * name) {
    if (model == nullptr || source == nullptr || source->buffer == nullptr || buft == nullptr) {
        return nullptr;
    }

    storage = std::make_unique<ggml_tensor>(*source);
    storage->buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, source));
    if (storage->buffer == nullptr) {
        storage.reset();
        return nullptr;
    }

    storage->data = ggml_backend_buffer_get_base(storage->buffer);
    storage->op = GGML_OP_NONE;
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        storage->src[j] = nullptr;
    }
    storage->view_src = nullptr;
    storage->view_offs = 0;
    storage->extra = nullptr;
    ggml_set_name(storage.get(), name);
    ggml_backend_buffer_set_usage(storage->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_copy(source, storage.get());

    model->bufs.push_back(storage->buffer);
    return storage.get();
}

bool llama_model_share_qwen4exp_mtp_tensors(llama_model * draft_model, const llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr) {
        return false;
    }
    if (draft_model->arch != LLM_ARCH_QWEN4EXP) {
        return true;
    }
    if (draft_model->tok_embd != nullptr && draft_model->output != nullptr) {
        return true;
    }
    if (target_model->arch != LLM_ARCH_QWEN4EXP) {
        return false;
    }

    const int64_t n_embd  = draft_model->hparams.n_embd;
    const int64_t n_vocab = draft_model->hparams.n_vocab;

    if (draft_model->tok_embd == nullptr) {
        ggml_tensor * tok_embd = target_model->tok_embd;
        if (tok_embd == nullptr ||
                tok_embd->ne[0] != n_embd || tok_embd->ne[1] != n_vocab) {
            return false;
        }
        if (llama_model_qwen4exp_io_needs_clone(tok_embd, draft_model->buft_input.buft)) {
            tok_embd = llama_model_clone_qwen4exp_io_tensor(
                    draft_model, tok_embd, draft_model->buft_input.buft,
                    draft_model->qwen4exp_tok_embd_ptr, "qwen4exp_tok_embd");
            if (tok_embd == nullptr) {
                return false;
            }
        }
        draft_model->tok_embd = tok_embd;
    }

    if (draft_model->output == nullptr) {
        ggml_tensor * output = target_model->output;
        if (output == nullptr ||
                output->ne[0] != n_embd || output->ne[1] != n_vocab) {
            return false;
        }
        if (llama_model_qwen4exp_io_needs_clone(output, draft_model->buft_output.buft)) {
            output = llama_model_clone_qwen4exp_io_tensor(
                    draft_model, output, draft_model->buft_output.buft,
                    draft_model->qwen4exp_output_ptr, "qwen4exp_output");
            if (output == nullptr) {
                return false;
            }
        }
        draft_model->output = output;
    }

    return draft_model->tok_embd != nullptr && draft_model->output != nullptr;
}
