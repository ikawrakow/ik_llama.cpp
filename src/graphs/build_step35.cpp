#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_step35() {
    ggml_cgraph * gf = new_graph_custom();
    ggml_tensor * cur;
    auto inp_pos     = build_inp_pos();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ASSERT(model.mtp && hparams.nextn_predict_layers > 0);
        GGML_ASSERT(batch.token && "Step35 MTP requires token batches");

        const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
            ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;
        const int n_heads_model = (int) hparams.nextn_predict_layers;
        const int n_heads = lctx.mtp_n_heads > 0
            ? std::max(1, std::min((int) lctx.mtp_n_heads, n_heads_model)) : n_heads_model;
        const int step = std::max(0, std::min((int) lctx.mtp_step_idx, n_heads - 1));
        const int il = n_layer_base + step;

        ggml_tensor * hidden_states = build_inp_mtp_states(n_embd);

        const bool step_independent_warmup = model.arch == LLM_ARCH_STEP35 &&
            (cparams.mtp_op_type == MTP_OP_WARMUP ||
             cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) && n_heads > 1;
        if (step_independent_warmup) {
            for (int i = n_heads - 1; i >= 0; --i) {
                const int head_il = n_layer_base + i;
                const bool is_first = i == 0;
                const bool emit_logits = is_first && cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
                cur = build_step35_mtp(model.layers[head_il], hidden_states, gf, inp_pos,
                        /*reduce_output=*/is_first, emit_logits);
                ggml_build_forward_expand(gf, cur);
            }
            return gf;
        }

        const bool reduce_mtp_output = cparams.mtp_op_type != MTP_OP_NONE;
        const bool emit_mtp_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
            cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        cur = build_step35_mtp(model.layers[il], hidden_states, gf, inp_pos,
                reduce_mtp_output, emit_mtp_logits);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();
    auto KQ_mask_swa = build_inp_KQ_mask_swa();
    //const float kq_scale = 1.0f / sqrtf(float(n_rot));
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
        ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;

    for (int il = 0; il < n_layer_base; ++il) {
        bool is_swa = hparams.swa_layers[il];
        auto & layer = const_cast<llama_layer&>(model.layers[il]);

        ggml_tensor * rope_factors = nullptr;
        const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
        if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
            rope_factors = build_rope_factors(il);
        }
        auto rope_freqs = layer.rope_freqs;
        layer.rope_freqs = nullptr;
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer_base - 1 && n_tokens > 1 && !cparams.mtp ? inp_out_ids : nullptr,
                rope_factors, is_swa ? KQ_mask_swa : KQ_mask, nullptr, nullptr, kq_scale, 0.0f, is_swa ? hparams.n_swa : 0,
                il, true, false, true);
        layer.rope_freqs = rope_freqs;

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            const bool  norm_w  = hparams.expert_weights_norm;
            const float w_scale = hparams.expert_weights_scale;
            const bool  scale_w = w_scale != 0.0f;
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, norm_w, scale_w, w_scale,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    //(llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    if (cparams.mtp) {
        ggml_tensor * mtp_embd = inpL->type == GGML_TYPE_F32 ? inpL : ggml_cast(ctx0, inpL, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);

        if (inp_out_ids) {
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * llm_build_context::build_step35_mtp(
        const llama_layer & mtp_layer,
        ggml_tensor * hidden_states_from_main_model,
        ggml_cgraph * gf,
        ggml_tensor * inp_pos,
        bool reduce_output,
        bool emit_logits,
        ggml_tensor ** hidden_out) {
    const int il = (int) (&mtp_layer - model.layers.data());

    GGML_ASSERT(mtp_layer.nextn.eh_proj && mtp_layer.nextn.enorm && mtp_layer.nextn.hnorm);
    GGML_ASSERT(mtp_layer.wq && mtp_layer.wk && mtp_layer.wv && mtp_layer.wo);

    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;
    ggml_tensor * tok_embd_w = mtp_layer.nextn.embed_tokens ? mtp_layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = build_inp_embd_mtp(tok_embd_w);
    ggml_tensor * cur = build_mtp_input(mtp_layer, hidden_states_from_main_model,
            tok_embd, il, "mtp_eh_proj");

    const bool is_swa = hparams.swa_layers[il];
    ggml_tensor * rope_factors = nullptr;
    const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
    if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
        rope_factors = build_rope_factors(il);
    }
    auto KQ_mask = is_swa ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k(il)));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur, inp_pos, nullptr,
            rope_factors, KQ_mask, nullptr, nullptr, kq_scale, 0.0f,
            is_swa ? hparams.n_swa : 0, il, true, false, true, false, false, nullptr, il);

    if (mtp_layer.ffn_gate_inp == nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up, nullptr, nullptr,
                mtp_layer.ffn_gate, nullptr, nullptr,
                mtp_layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
    } else {
        cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_gate_inp, nullptr,
                mtp_layer.ffn_up_exps, nullptr,
                mtp_layer.ffn_gate_exps, nullptr,
                mtp_layer.ffn_down_exps, nullptr,
                mtp_layer.ffn_exp_probs_b,
                mtp_layer.ffn_up_shexp, nullptr,
                mtp_layer.ffn_gate_shexp, nullptr,
                mtp_layer.ffn_down_shexp, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_post_ffn", il);
    if (hidden_out) {
        *hidden_out = cur;
    }

    ggml_tensor * output_hidden = cur;
    if (reduce_output) {
        if (cparams.mtp_op_type != MTP_OP_NONE && n_tokens > 1) {
            output_hidden = ggml_view_2d(ctx0, cur, n_embd, 1,
                    cur->nb[1], (size_t) (n_tokens - 1) * cur->nb[1]);
        } else if (inp_out_ids) {
            output_hidden = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }
    if (reduce_output) {
        ggml_tensor * mtp_embd = output_hidden->type == GGML_TYPE_F32 ? output_hidden : ggml_cast(ctx0, output_hidden, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);
    }

    if (!emit_logits) {
        return output_hidden;
    }

    ggml_tensor * head_norm = mtp_layer.nextn.shared_head_norm
        ? mtp_layer.nextn.shared_head_norm : model.output_norm;
    ggml_tensor * head = mtp_layer.nextn.shared_head_head
        ? mtp_layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_norm && head);
    cur = llm_build_context::build_output(lctx, ctx0, output_hidden, head, head_norm, cb);
    cb(cur, "result_output", -1);
    return cur;
}
