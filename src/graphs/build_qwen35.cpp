#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

ggml_cgraph * llm_build_context::build_qwen35moe() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * cur = nullptr;

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        ggml_tensor * hidden_states_from_main_model = build_inp_mtp_states(hparams.n_embd);

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        const bool emit_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
                cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        cur = build_qwen35moe_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos, emit_logits);
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens &&
                (!lctx.cparams.mtp || n_outputs > 0)) ? build_inp_out_ids() : nullptr;
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 && !lctx.cparams.mtp ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos,
                        il == n_transformer_layers - 1 && !lctx.cparams.mtp ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
            }

            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    nullptr,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps, nullptr, model.layers[il].ffn_gate_inp_shexp);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        if (n_outputs == 0) {
            cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
            cb(cur, "result_norm", -1);
        } else if (lctx.cparams.mtp) {
            auto hidden = llm_build_norm(ctx0, inpL, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
            cb(hidden, "result_norm", -1);
            // MTP reads this after graph compute; do not recycle its buffer
            ggml_set_output(hidden);
            if (inp_out_ids && hidden->ne[1] > n_outputs) {
                inpL = ggml_get_rows(ctx0, hidden, inp_out_ids);
            } else {
                inpL = hidden;
            }
            cur = build_output(lctx, ctx0, inpL, model.output, nullptr, cb);
            cb(cur, "result_output", -1);
        } else {
            cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
            cb(cur, "result_output", -1);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen35() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // MTP tail-only graph
        ggml_tensor * hidden_states_from_main_model = build_inp_mtp_states(hparams.n_embd);

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        const bool emit_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
                cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        cur = build_qwen35_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos, emit_logits);
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens &&
                (!lctx.cparams.mtp || n_outputs > 0)) ? build_inp_out_ids() : nullptr;
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = nullptr;

        // SenseNova-Loop: mHC residual streams over the looped layers
        const bool use_mhc = hparams.mhc_num_stream >= 2 && hparams.mhc_end > hparams.mhc_start;
        const int64_t S = hparams.mhc_num_stream;
        const int mhc_iters = hparams.dsv4_hc_sinkhorn_iters;

        auto mhc_apply = [&](ggml_tensor * y, ggml_tensor * R,
                             ggml_tensor * post, ggml_tensor * comb) -> ggml_tensor * {
            return build_mhc_post(y, post, R, comb, n_embd, S, true);
        };

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            const uint32_t ip = hparams.phys_layer(il);
            const bool in_mhc = use_mhc && ip >= hparams.mhc_start && ip < hparams.mhc_end;
            const bool mhc_enter = in_mhc && ip == hparams.mhc_start;

            ggml_tensor * Rin = inpL;
            if (mhc_enter) {
                Rin = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
                                  ggml_new_tensor_3d(ctx0, inpL->type, n_embd, S, n_tokens));
                cb(Rin, "mhc_R_entry", il);
            }

            ggml_tensor * out_ids = (!in_mhc && il == n_transformer_layers - 1 && !lctx.cparams.mtp) ? inp_out_ids : nullptr;

            if (in_mhc) {
                GGML_ASSERT(model.layers[il].mhc_attn_w != nullptr && model.layers[il].mhc_ffn_w != nullptr);

                ggml_tensor * post = nullptr;
                ggml_tensor * comb = nullptr;
                ggml_tensor * x = build_mhc_pre_weight_bias(Rin, model.layers[il].mhc_attn_w,
                        model.layers[il].mhc_attn_b, S, mhc_iters, &post, &comb, hparams.mhc_pre_scale);
                ggml_tensor * y;
                if (hparams.is_recurrent(il)) {
                    y = delta.build_layer_attn_linear(ctx0, gf, x, nullptr, il, cb, true);
                } else {
                    y = build_std_attention(gf, model.layers[il].attn_norm, x, inp_pos,
                            nullptr, nullptr, KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il,
                            true, false, false, false, true);
                }
                Rin = mhc_apply(y, Rin, post, comb);

                x = build_mhc_pre_weight_bias(Rin, model.layers[il].mhc_ffn_w,
                        model.layers[il].mhc_ffn_b, S, mhc_iters, &post, &comb, hparams.mhc_pre_scale);
                y = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, x,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false, false);
                Rin = mhc_apply(y, Rin, post, comb);

                if (ip + 1 == hparams.mhc_end) {
                    // mean of the residual streams when leaving the region
                    ggml_tensor * mean = ggml_sum_rows_ext(ctx0, Rin, 1);
                    cur = ggml_reshape_2d(ctx0, ggml_scale(ctx0, mean, 1.0f/(float) S), n_embd, n_tokens);
                    cb(cur, "mhc_R_exit", il);
                } else {
                    cur = Rin;
                }
            } else {
                if (hparams.is_recurrent(il)) {
                    cur = delta.build_layer_attn_linear(ctx0, gf, inpL, out_ids, il, cb);
                } else {
                    cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos,
                            out_ids, nullptr,
                            KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
                }

                cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
            }

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        // reduce output tokens if the last layer is looped
        if (inp_out_ids && !lctx.cparams.mtp && use_mhc && n_transformer_layers > 0) {
            const uint32_t ip_last = hparams.phys_layer(n_transformer_layers - 1);
            if (ip_last >= hparams.mhc_start && ip_last < hparams.mhc_end) {
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
                cur = inpL;
            }
        }

        if (n_outputs == 0) {
            cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
            cb(cur, "result_norm", -1);
        } else if (lctx.cparams.mtp) {
            auto hidden = llm_build_norm(ctx0, inpL, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
            cb(hidden, "result_norm", -1);
            // MTP reads this after graph compute; do not recycle its buffer
            ggml_set_output(hidden);
            if (inp_out_ids && hidden->ne[1] > n_outputs) {
                inpL = ggml_get_rows(ctx0, hidden, inp_out_ids);
            } else {
                inpL = hidden;
            }
            cur = build_output(lctx, ctx0, inpL, model.output, nullptr, cb);
            cb(cur, "result_output", -1);
        } else {
            cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
            cb(cur, "result_output", -1);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_tensor * llm_build_context::build_qwen35moe_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos,
    bool emit_logits) {

    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    ggml_tensor * cur = build_mtp_input(mtp_layer, prev_embeddings, token_emb, il, "mtp_fused");

    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!mtp_layer.wq || !mtp_layer.wk || !mtp_layer.wv || !mtp_layer.wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wq=%p, wk=%p, wv=%p, wo=%p)\n",
                __func__, il, (void*)mtp_layer.wq, (void*)mtp_layer.wk,
                (void*)mtp_layer.wv, (void*)mtp_layer.wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, nullptr, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr);

    if (!emit_logits) {
        return cur;
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
            mtp_layer.ffn_gate_inp,  nullptr,
            mtp_layer.ffn_up_exps,   nullptr,
            mtp_layer.ffn_gate_exps, nullptr,
            mtp_layer.ffn_down_exps, nullptr,
            nullptr,
            mtp_layer.ffn_up_shexp,    nullptr,
            mtp_layer.ffn_gate_shexp,  nullptr,
            mtp_layer.ffn_down_shexp,  nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true, false, 0.0f,
            LLM_EXPERT_GATING_FUNC_SOFTMAX,
            LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps, nullptr, mtp_layer.ffn_gate_inp_shexp);

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    if (mtp_layer.nextn.shared_head_norm != nullptr) {
        cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, cb, -1);
    }
    cb(cur, "result_norm", -1);
    // MTP reads this after graph compute (chained drafts); do not recycle its buffer
    ggml_set_output(cur);

    cur = build_output(lctx, ctx0, cur, model.output_mtp, cb);
    cb(cur, "result_output", -1);

    return cur;
}

struct ggml_tensor * llm_build_context::build_qwen35_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos,
    bool emit_logits) {

    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    ggml_tensor * cur = build_mtp_input(mtp_layer, prev_embeddings, token_emb, il, "mtp_fused");

    // Self-Attention (wq may be shared from main model's last layer)
    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!model.layers[il].wq || !model.layers[il].wk || !model.layers[il].wv || !model.layers[il].wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wq=%p, wk=%p, wv=%p, wo=%p)\n",
                __func__, il, (void*)model.layers[il].wq, (void*)model.layers[il].wk,
                (void*)model.layers[il].wv, (void*)model.layers[il].wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, inp_out_ids, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr);

    if (!emit_logits) {
        return cur;
    }

    // Dense FFN — optional (9B and 4B don't have FFN in MTP layer)
    if (mtp_layer.ffn_gate != nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up,   NULL, NULL,
                mtp_layer.ffn_gate, NULL, NULL,
                mtp_layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    if (mtp_layer.nextn.shared_head_norm != nullptr) {
        cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, cb, -1);
    }
    cb(cur, "result_norm", -1);
    // MTP reads this after graph compute (chained drafts); do not recycle its buffer
    ggml_set_output(cur);

    cur = build_output(lctx, ctx0, cur, model.output_mtp, cb);
    cb(cur, "result_output", -1);

    return cur;
}
