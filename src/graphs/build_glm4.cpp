#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_glm4_moe() {
    // create a new graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;

    // position embeddings
    struct ggml_tensor * inp_pos = build_inp_pos();

    auto rope_cache = model.split_mode != LLAMA_SPLIT_MODE_GRAPH && cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        ggml_tensor* hidden_states_from_main_model;

        if (cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        } else {
            hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_embd);
        }
        ggml_set_name(hidden_states_from_main_model, "result_embd_pooled");
        ggml_set_input(hidden_states_from_main_model);

        lctx.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_mtp_tail(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos, rope_cache);

    } else {
        struct ggml_tensor * inpL;

        // input embeddings
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

        struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

        // output token IDs (for last layer cropping)
        struct ggml_tensor * inp_out_ids = (n_tokens > 1 && !lctx.cparams.mtp) ? build_inp_out_ids() : nullptr;

        float kq_scale = 1.0f/sqrtf(float(n_embd_head));

        // Only process up to last layer (skip final NextN layer)
        // Final layer tensors are loaded but not processed in forward pass
        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // self-attention
            if (rope_cache == nullptr) {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                        inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, kq_scale, 0.0f, 0, il, true, false, true);
            } else {
                // Pre-attention norm
                cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(cur, "attn_norm", il);

                auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                        model.layers[il].wqkv, model.layers[il].bqkv,
                        model.layers[il].wqk, model.layers[il].bqk,
                        model.layers[il].wq, model.layers[il].bq,
                        model.layers[il].wk, model.layers[il].bk,
                        model.layers[il].wv, model.layers[il].bv,
                        model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.f, il);

                // apply RoPE
                if (rope_cache) {
                    Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                    Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
                } else {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // build attention KV (no unified cache)
                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask,
                        n_tokens, kv_head, n_kv,
                        1.0f/sqrtf(float(n_embd_head)), cb, il);

                if (il == n_transformer_layers - 1 && inp_out_ids) {
                    // skip computing output for unused tokens
                    cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                    if (rope_cache) {
                        inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
                    }
                }
            }

            // crop output on last layer

            // residual connection for attention output
            ggml_tensor * ffn_inp;
            if (rope_cache) {
                ffn_inp = ggml_add(ctx0, cur, inpSA);
                cb(ffn_inp, "ffn_inp", il);
            } else {
                ffn_inp = cur;
            }

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                // dense FFN
                cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
                cb(cur, "ffn_out", il);
            } else {
                cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                        model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                        model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                        model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                        model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                        model.layers[il].ffn_exp_probs_b,
                        model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                        model.layers[il].ffn_gate_shexp,  nullptr,
                        model.layers[il].ffn_down_shexp,  nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
            }

            // residual and context vector
            //cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // prepare next layer input
            inpL = cur;
        }
        cur = inpL;

        // lm head
        cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
        cb(cur, "result_output", -1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_glm4() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv == nullptr) {
                Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                }
                Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                }
                Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                }
            } else {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);
                if (model.layers[il].bqkv) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }
                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Post-attention norm (new!)
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "post_attn_norm", il);

        // Add the input (residual connection after post-attention norm)
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            // MLP
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up, NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

            // Post-MLP norm
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "post_mlp_norm", il);
        }

        // Add residual connection after post-MLP norm
        inpL = ggml_add(ctx0, cur, ffn_inp);
        cb(inpL, "l_out", il);
    }

    // Final norm
    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // Output projection
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_tensor * llm_build_context::build_mtp_tail(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos,
    struct ggml_tensor * rope_cache) {
    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    struct ggml_tensor * inp_out_ids = build_inp_out_ids();

    // If nextn.embed_tokens is missing (GLM-4.6), use model.tok_embd
    ggml_tensor * mtp_embd_weights = mtp_layer.nextn.embed_tokens;
    if (mtp_embd_weights == nullptr) {
        mtp_embd_weights = model.tok_embd;
    }
    ggml_tensor * token_emb = build_inp_embd_mtp(mtp_embd_weights);

    ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
    cb(combined, "mtp_concat", il);
    ggml_tensor* cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);

    // Self-Attention
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    ggml_tensor * ffn_inp;
    if (rope_cache == nullptr) {
        cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
                inp_pos, nullptr, nullptr,
                KQ_mask, nullptr, nullptr,
                kq_scale, 0.0f, 0, il, true, false, true, false, false, nullptr);
        ffn_inp = cur;
    } else {
        struct ggml_tensor * inpSA = cur;
        cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
        auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                nullptr, nullptr,
                nullptr, nullptr,
                mtp_layer.wq, mtp_layer.bq,
                mtp_layer.wk, mtp_layer.bk,
                mtp_layer.wv, mtp_layer.bv,
                mtp_layer.attn_q_norm, mtp_layer.attn_k_norm,
                0.f, il);
        Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
        Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        mtp_layer.wo, NULL,
                        Kcur, Vcur, Qcur, KQ_mask,
                        n_tokens, kv_head, n_kv,
                        kq_scale, cb, il);
        ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "mtp_ffn_inp", il);
    }

    // FFN
    cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, ffn_inp,
            mtp_layer.ffn_gate_inp,  NULL,
            mtp_layer.ffn_up_exps,   NULL,
            mtp_layer.ffn_gate_exps, NULL,
            mtp_layer.ffn_down_exps, NULL,
            mtp_layer.ffn_exp_probs_b,
            mtp_layer.ffn_up_shexp,    nullptr,
            mtp_layer.ffn_gate_shexp,  nullptr,
            mtp_layer.ffn_down_shexp,  nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps);

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // If nextn.shared_head_head is missing (GLM-4.6), use model.output (Main LM Head)
    ggml_tensor * mtp_head_weights = mtp_layer.nextn.shared_head_head;
    if (mtp_head_weights == nullptr) {
        mtp_head_weights = model.output;
    }
    cur = llm_build_lora_mm(lctx, ctx0, mtp_head_weights, cur);
    cb(cur, "result_output", -1);

    return cur;
}
