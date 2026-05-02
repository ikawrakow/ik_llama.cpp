#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_llama() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inp_attn_scale = nullptr;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    if (model.arch == LLM_ARCH_LLAMA4) {
        inp_attn_scale = build_input_scale(n_tokens);
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    //bool is_swa = hparams.n_swa > 0 && h_params.n_swa_pattern > 0 ?
    ggml_tensor * KQ_mask = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = nullptr;
    if (hparams.n_swa > 0 && hparams.n_swa_pattern > 0) {
        KQ_mask_swa = build_inp_KQ_mask_swa();
    }

    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    //const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : 1.f;
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        bool use_rope = model.arch == LLM_ARCH_LLAMA4 ? (il + 1) % hparams.n_no_rope_layer_step != 0 : true;
        auto this_KQ_mask = hparams.n_swa > 0 && hparams.n_swa_pattern > 0 && il % hparams.n_swa_pattern < (hparams.n_swa_pattern - 1) ?
            KQ_mask_swa : KQ_mask;
        int this_n_swa = this_KQ_mask == KQ_mask_swa ? hparams.n_swa : 0;

        // rope freq factors for llama3; may return nullptr for llama2 and other models
        //auto rope_factors = build_rope_factors(il);

        // self-attention
        if (use_rope) {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                    inp_pos, il == n_layer - 1 && n_tokens > 1 ? inp_out_ids : nullptr, nullptr,
                    this_KQ_mask, nullptr, nullptr, kq_scale, hparams.f_attention_scale, this_n_swa, il, true, false, true);
        }
        else {

            auto rope_factors = build_rope_factors(il);

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk, model.layers[il].bqk,
                    model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv,
                    nullptr, nullptr, hparams.f_attention_scale, il);

            if (use_rope) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            } else if (inp_attn_scale) {
                Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (model.arch == LLM_ARCH_LLAMA4 && use_rope && hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, this_KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il, nullptr,
                    this_n_swa);
        }

        if (il == n_layer - 1 && !use_rope && inp_out_ids) {
            // skip computing output for unused tokens
            auto inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            cb(cur, "last_attn", il);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            cb(inpSA, "last_ffn_inp", il);
        }

        // For Granite architecture
        if (hparams.f_residual_scale) {
            // Why is hparams.f_residual_scale not simply absorbed into model.layers[il].wv ?
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        ggml_tensor * ffn_inp;
        if (use_rope) {
            ffn_inp = cur;
        } else {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // non-MoE
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else if (model.arch == LLM_ARCH_LLAMA4) {
            // llama4 MoE
            ggml_tensor * ffn_inp_normed = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, ffn_inp_normed,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    cb, il, gf, true, model.layers[il].ffn_up_gate_exps);

            // Shared experts
            ggml_tensor * shexp_out = llm_build_ffn(ctx0, lctx, nullptr, ffn_inp_normed,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(shexp_out, "ffn_moe_shexp", il);

            cur = ggml_add(ctx0, moe_out, shexp_out);
            cb(cur, "ffn_moe_out_merged", il);

        } else {
            // MoE branch
            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf, true);
            cb(cur, "ffn_moe_out", il);
        }

        // For Granite architecture
        if (hparams.f_residual_scale) {
            // Why is hparams.f_residual_scale not simply absorbed into model.layers[il].ffn_down_exps ?
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        //cur = ggml_add(ctx0, cur, ffn_inp);
        //cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // lm_head
    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);

    // For Granite architecture
    if (hparams.f_logit_scale) {
        // Why is hparams.f_logit_scale not simply absorbed into model.output ?
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

